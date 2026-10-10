"""Pinned, unquantized BF16 teacher. One load serves an entire local batch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl


class LocalConsolidator:
    def __init__(self, config):
        from transformers import AutoModelForMultimodalLM, AutoTokenizer
        self.config = config
        if config.get('quantization') != 'none' or config.get('inference_dtype') != 'bfloat16':
            raise ValueError('The authorized consolidator requires unquantized BF16 inference')
        root = Path(config['model_path'])
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

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=1024):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, inputs['input_ids'].shape[1]:]
        return {'text': self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'prompt_tokens': inputs['input_ids'].shape[1], 'generated_tokens': len(generated),
                'hit_generation_limit': len(generated) >= max_new_tokens}


    @torch.inference_mode()
    def generate_batch(self, messages_batch, max_new_tokens=1024):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('A positive integer generation budget is required')
        if not messages_batch:
            return []
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False,
                 add_generation_prompt=True, enable_thinking=False) for messages in messages_batch]
        inputs = self.tokenizer(texts, add_special_tokens=False, padding=True,
                                padding_side='left', return_attention_mask=True,
                                return_tensors='pt').to(self.model.device)
        width = inputs['input_ids'].shape[1]
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=False)
        if output.ndim != 2 or output.shape[0] != len(texts) or not width <= output.shape[1] <= width + max_new_tokens:
            raise RuntimeError('Teacher generation must return one bounded response per prompt')
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        stops = {eos} if isinstance(eos, int) else set(eos or [])
        results = []
        for index, tokens in enumerate(output[:, width:]):
            end = next((i + 1 for i, token in enumerate(tokens.tolist()) if token in stops), len(tokens))
            generated = tokens[:end]
            results.append({'text': self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'prompt_tokens': int(inputs['attention_mask'][index].sum().item()),
                'generated_tokens': end, 'hit_generation_limit': end >= max_new_tokens})
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
    args = parser.parse_args()
    root = Path(args.output)
    if (root / 'manifest.json').exists():
        raise FileExistsError('Completed teacher batch exists')
    config = json.loads(Path(args.config).read_text())['search_consolidator']
    rows = load_jsonl(args.input)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Teacher query IDs must be unique')
    contract = {'input_sha256': digest(args.input), 'teacher': config,
                'max_new_tokens': args.max_new_tokens}
    arguments = vars(args).copy()
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
    started = time.monotonic()
    teacher = LocalConsolidator(config)
    atomic_json(root / 'deployment.json', teacher.identity)
    pending = [row for row in rows if row['id'] not in seen]
    reused = len(results)
    batches = 0
    with partial.open('a') as handle:
        for offset in range(0, len(pending), args.batch_size):
            group = pending[offset:offset + args.batch_size]
            generated = ([teacher.generate(group[0]['messages'], args.max_new_tokens)]
                         if args.batch_size == 1 else
                         teacher.generate_batch([row['messages'] for row in group], args.max_new_tokens))
            if len(generated) != len(group):
                raise RuntimeError('Teacher batch coverage differs from its queries')
            items = [{'id': row['id'], **response, 'teacher': teacher.identity}
                     for row, response in zip(group, generated, strict=True)]
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n'); handle.flush()
                results.append(item)
            batches += 1
            print(json.dumps({'generated': len(results), 'requested': len(rows),
                              'batch_size': len(group), 'seconds': time.monotonic() - started}), flush=True)
    write_jsonl(root / 'responses.jsonl', results)
    proof = {'examples': len(results), 'seconds': time.monotonic() - started,
             'teacher': teacher.identity, 'neural_teacher_inference_executed': True,
             'configured_batch_size': args.batch_size, 'responses_reused': reused,
             'new_teacher_queries': len(pending), 'generation_batches_this_invocation': batches,
             'generations_truncated': sum(r['hit_generation_limit'] for r in results),
             'semantic_validation': 'caller_must_validate_raw_responses'}
    atomic_json(root / 'manifest.json', manifest('local_teacher_inference', arguments,
                [args.config, args.input, Path(config['model_path']) / 'weights.manifest.json'],
                [root / 'responses.jsonl', root / 'deployment.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
