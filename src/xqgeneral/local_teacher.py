"""Pinned, unquantized BF16 teacher. One load serves an entire local batch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest, write_jsonl
from .sampling import SAMPLING_KEYS, checked_seed, generation_software, sample_rng
from .teacher_generation import (SEED_PROFILE, batch_seed, checked_budget, checked_prompt_width,
                                 checked_response, generation_profile)


class LocalConsolidator:
    def __init__(self, config):
        self.generation_profile = (generation_profile(config['generation_profile'])
                                   if config.get('generation_profile') is not None else None)
        self.max_context_tokens = config.get('max_context_tokens')
        if self.generation_profile is not None:
            if self.max_context_tokens != self.generation_profile['max_context_tokens']:
                raise ValueError('Teacher context differs from its explicit generation profile')
        if self.max_context_tokens is not None:
            checked_budget(1, self.max_context_tokens)
        from transformers import AutoModelForMultimodalLM, AutoTokenizer
        self.config = config
        if config.get('quantization') != 'none' or config.get('inference_dtype') != 'bfloat16':
            raise ValueError('The authorized consolidator requires unquantized BF16 inference')
        root = Path(config['model_path'])
        model_config = json.loads((root / 'config.json').read_text())
        capacity = model_config.get('text_config', model_config).get('max_position_embeddings')
        if self.max_context_tokens is not None and (type(capacity) is not int or
                capacity < self.max_context_tokens):
            raise ValueError('Teacher context exceeds the pinned model capacity')
        proof = json.loads((root / 'weights.manifest.json').read_text())
        if (proof['repository'] != config['repository'] or proof['revision'] != config['revision'] or
                proof['quantization'] != 'none' or not proof['all_official_lfs_sha256_matched']):
            raise ValueError('Full teacher weights must pass official hash verification before loading')
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForMultimodalLM.from_pretrained(
            root, local_files_only=True, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa')
        self.model.eval()
        if (getattr(self.model, 'is_loaded_in_4bit', False) or getattr(self.model, 'is_loaded_in_8bit', False) or
                getattr(self.model.config, 'quantization_config', None)):
            raise ValueError('Loaded teacher unexpectedly uses quantization')
        dtypes = Counter()
        for parameter in self.model.parameters():
            if not parameter.is_floating_point():
                raise ValueError('Loaded teacher contains a non-floating parameter')
            dtypes[str(parameter.dtype)] += parameter.numel()
        if not dtypes.get('torch.bfloat16'):
            raise ValueError('BF16 teacher parameters are missing')
        self.identity = {'repository': config['repository'], 'revision': config['revision'],
                         'quantization': 'none', 'inference_dtype': 'bfloat16',
                         'parameter_elements_by_dtype': dict(dtypes),
                         'weight_manifest_sha256': digest(root / 'weights.manifest.json'),
                         'visible_cuda_devices': torch.cuda.device_count(),
                         'allocated_gib_by_device': [torch.cuda.memory_allocated(i) / 2**30
                                                    for i in range(torch.cuda.device_count())]}
        if self.max_context_tokens is not None:
            self.identity.update(max_context_tokens=self.max_context_tokens, model_context_capacity=capacity)
        if self.generation_profile is not None:
            self.identity.update(generation_profile=self.generation_profile,
                                 generation_software=generation_software())

    def chat_options(self):
        profile = getattr(self, 'generation_profile', None)
        return ({'reasoning_effort': profile['reasoning_effort']} if profile else {'enable_thinking': False})

    def decoding_options(self):
        profile = getattr(self, 'generation_profile', None)
        return {k: profile[k] for k in SAMPLING_KEYS} if profile else {'do_sample': False}

    def check_budget(self, budget):
        checked_budget(budget, getattr(self, 'max_context_tokens', None))
        profile = getattr(self, 'generation_profile', None)
        if profile is not None and budget > profile['max_new_tokens']:
            raise ValueError('Teacher output budget exceeds its generation profile')

    def response(self, generated, prompt_tokens, width, budget, requested):
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        result = {'text': raw.strip(), 'prompt_tokens': prompt_tokens,
                  'generated_tokens': len(generated), 'hit_generation_limit': len(generated) >= budget}
        if getattr(self, 'max_context_tokens', None) is not None:
            result.update(padded_prompt_tokens=width, effective_max_new_tokens=budget,
                          context_limited_output=budget < requested)
        if getattr(self, 'generation_profile', None) is not None:
            completed = '</think>' in raw
            result.update(raw_text=raw, reasoning_completed=completed,
                          text=raw.split('</think>', 1)[1].strip() if completed else raw.strip())
        return result

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=1024):
        self.check_budget(max_new_tokens)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.chat_options())
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        width = inputs['input_ids'].shape[1]
        budget = checked_prompt_width(width, max_new_tokens, getattr(self, 'max_context_tokens', None))
        inputs = inputs.to(self.model.device)
        output = self.model.generate(**inputs, **self.decoding_options(), max_new_tokens=budget,
                                      pad_token_id=self.tokenizer.eos_token_id)
        if output.ndim != 2 or output.shape[0] != 1 or not width <= output.shape[1] <= width + budget:
            raise RuntimeError('Teacher generation must return one bounded response per prompt')
        return self.response(output[0, width:], width, width, budget, max_new_tokens)


    @torch.inference_mode()
    def generate_batch(self, messages_batch, max_new_tokens=1024):
        self.check_budget(max_new_tokens)
        if not messages_batch:
            return []
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False,
                 add_generation_prompt=True, **self.chat_options()) for messages in messages_batch]
        inputs = self.tokenizer(texts, add_special_tokens=False, padding=True,
                                padding_side='left', return_attention_mask=True,
                                return_tensors='pt')
        lengths = [int(x) for x in inputs['attention_mask'].sum(dim=1).tolist()]
        groups = {}
        for index, length in enumerate(lengths):
            budget = checked_prompt_width(length, max_new_tokens, getattr(self, 'max_context_tokens', None))
            groups.setdefault(budget, []).append(index)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        stops = {eos} if isinstance(eos, int) else set(eos or [])
        results = [None] * len(texts)
        for budget, indices in groups.items():
            width = max(lengths[i] for i in indices)
            group = inputs.__class__({key: value[indices, -width:] for key, value in inputs.items()})
            group = group.to(self.model.device)
            output = self.model.generate(**group, **self.decoding_options(), max_new_tokens=budget,
                pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=False)
            if output.ndim != 2 or output.shape[0] != len(indices) or not width <= output.shape[1] <= width + budget:
                raise RuntimeError('Teacher generation must return one bounded response per prompt')
            for index, tokens in zip(indices, output[:, width:], strict=True):
                end = next((i + 1 for i, token in enumerate(tokens.tolist()) if token in stops), len(tokens))
                results[index] = self.response(tokens[:end], lengths[index], width, budget, max_new_tokens)
        return results


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Use a positive integer')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/teachers.json')
    parser.add_argument('--input', required=True, help='JSONL with id and messages')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=positive_int, default=1024)
    parser.add_argument('--batch-size', type=positive_int, default=1)
    parser.add_argument('--max-context-tokens', type=positive_int)
    parser.add_argument('--generation-config')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    root = Path(args.output)
    if (root / 'manifest.json').exists():
        raise FileExistsError('Completed teacher batch exists')
    config = dict(json.loads(Path(args.config).read_text())['search_consolidator'])
    if config.get('generation_profile') is not None:
        raise ValueError('Supply teacher generation settings through an explicit generation-config file')
    profile = generation_profile(json.loads(Path(args.generation_config).read_text())) if args.generation_config else None
    if profile is not None:
        checked_seed(args.seed)
        if args.max_new_tokens != profile['max_new_tokens'] or (args.max_context_tokens is not None and
                args.max_context_tokens != profile['max_context_tokens']):
            raise ValueError('Teacher CLI lengths differ from the generation profile')
        args.max_context_tokens = profile['max_context_tokens']
        config['generation_profile'] = profile
    elif args.seed is not None:
        raise ValueError('An explicit teacher seed requires a stochastic generation profile')
    if args.max_context_tokens is not None:
        config['max_context_tokens'] = args.max_context_tokens
    checked_budget(args.max_new_tokens, config.get('max_context_tokens'))
    rows = load_jsonl(args.input)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Teacher query IDs must be unique')
    contract = {'input_sha256': digest(args.input), 'teacher': config,
                'max_new_tokens': args.max_new_tokens}
    arguments = vars(args).copy()
    for key in ['max_context_tokens', 'generation_config', 'seed']:
        if arguments[key] is None:
            arguments.pop(key)
    if profile is not None:
        contract.update(generation_config_sha256=digest(args.generation_config),
                        seed=args.seed, seed_profile=SEED_PROFILE, generation_software=generation_software(),
                        code=code_identity())
    if args.batch_size != 1:
        contract['batch_size'] = args.batch_size
    else:
        arguments.pop('batch_size')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'contract.json').exists():
        if json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Teacher continuation inputs changed')
    else:
        atomic_json(root / 'contract.json', contract)
    partial = root / 'responses.partial.jsonl'
    results = load_jsonl(partial) if partial.exists() else []
    seen = {r['id'] for r in results}
    if len(seen) != len(results) or not seen <= {r['id'] for r in rows}:
        raise ValueError('Teacher continuation has duplicate or foreign IDs')
    for item in results:
        checked_response(item, args.max_new_tokens, config.get('max_context_tokens'), profile)
        if profile is not None and (item['id'] not in item.get('generation_batch_ids', []) or
                not set(item.get('generation_batch_ids', [])) <= {r['id'] for r in rows} or
                item.get('sampling_seed') != batch_seed(args.seed, item.get('generation_batch_ids', []))):
            raise ValueError('Teacher continuation sampling identities changed')
    started = time.monotonic()
    teacher = LocalConsolidator(config)
    atomic_json(root / 'deployment.json', teacher.identity)
    pending = [row for row in rows if row['id'] not in seen]
    reused = len(results)
    batches = 0
    with partial.open('a') as handle:
        for offset in range(0, len(pending), args.batch_size):
            group = pending[offset:offset + args.batch_size]
            sampling_seed = batch_seed(args.seed, [row['id'] for row in group]) if profile else None
            with sample_rng(teacher.model.device if profile else 'cpu', sampling_seed):
                generated = ([teacher.generate(group[0]['messages'], args.max_new_tokens)]
                             if args.batch_size == 1 else
                             teacher.generate_batch([row['messages'] for row in group], args.max_new_tokens))
            if len(generated) != len(group):
                raise RuntimeError('Teacher batch coverage differs from its queries')
            for response in generated:
                checked_response(response, args.max_new_tokens, config.get('max_context_tokens'), profile)
            items = [{'id': row['id'], **response, 'teacher': teacher.identity}
                     for row, response in zip(group, generated, strict=True)]
            if profile is not None:
                for item in items:
                    item.update(sampling_seed=sampling_seed, generation_batch_ids=[row['id'] for row in group])
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n'); handle.flush()
                results.append(item)
            batches += 1
            print(json.dumps({'generated': len(results), 'requested': len(rows),
                              'batch_size': len(group), 'seconds': time.monotonic() - started}), flush=True)
    if (digest(args.input) != contract['input_sha256'] or profile is not None and
            (digest(args.generation_config) != contract['generation_config_sha256'] or
             code_identity() != contract['code'])):
        raise ValueError('Teacher inputs or frozen generation execution changed during inference')
    write_jsonl(root / 'responses.jsonl', results)
    proof = {'examples': len(results), 'seconds': time.monotonic() - started,
             'teacher': teacher.identity, 'neural_teacher_inference_executed': True,
             'configured_batch_size': args.batch_size, 'responses_reused': reused,
             'new_teacher_queries': len(pending), 'generation_batches_this_invocation': batches,
             'generations_truncated': sum(r['hit_generation_limit'] for r in results),
             'semantic_validation': 'caller_must_validate_raw_responses'}
    if config.get('max_context_tokens') is not None:
        proof.update(max_context_tokens=config['max_context_tokens'], max_new_tokens=args.max_new_tokens,
                     all_prompts_and_outputs_bounded=True, prompts_truncated=False,
                     context_limited_responses=sum(r['context_limited_output'] for r in results))
    if profile is not None:
        proof.update(generation_profile=profile, seed_profile=SEED_PROFILE, reasoning_raw_text_preserved=True,
                     reasoning_completed=sum(r['reasoning_completed'] for r in results))
    input_paths = [args.config, args.input, Path(config['model_path']) / 'weights.manifest.json']
    if args.generation_config:
        input_paths.append(args.generation_config)
    atomic_json(root / 'manifest.json', manifest('local_teacher_inference', arguments,
                input_paths,
                [root / 'responses.jsonl', root / 'deployment.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
