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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/teachers.json')
    parser.add_argument('--input', required=True, help='JSONL with id and messages')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
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
    with partial.open('a') as handle:
        for row in rows:
            if row['id'] in seen:
                continue
            item = {'id': row['id'], **teacher.generate(row['messages'], args.max_new_tokens),
                    'teacher': teacher.identity}
            handle.write(json.dumps(item, ensure_ascii=False) + '\n'); handle.flush()
            results.append(item)
            print(json.dumps({'generated': len(results), 'requested': len(rows),
                              'seconds': time.monotonic() - started}), flush=True)
    write_jsonl(root / 'responses.jsonl', results)
    proof = {'examples': len(results), 'seconds': time.monotonic() - started,
             'teacher': teacher.identity, 'neural_teacher_inference_executed': True,
             'generations_truncated': sum(r['hit_generation_limit'] for r in results),
             'semantic_validation': 'caller_must_validate_raw_responses'}
    atomic_json(root / 'manifest.json', manifest('local_teacher_inference', vars(args),
                [args.config, args.input, Path(config['model_path']) / 'weights.manifest.json'],
                [root / 'responses.jsonl', root / 'deployment.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
