"""Optional, bounded vLLM inference for the pinned full BF16 teacher."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .evidence import digest
from .foundation_handoff import Artifacts
from .local_teacher import LocalConsolidator
from .sampling import checked_seed
from .teacher_generation import checked_budget, checked_prompt_width, generation_profile

REQUEST_SEED_PROFILE = 'sha256_teacher_batch_seed_and_request_ordinal_v1'
MAX_SEQUENCES = 16


def request_seed(seed, ordinal):
    checked_seed(seed)
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('Teacher request ordinal must be a nonnegative integer')
    raw = json.dumps([REQUEST_SEED_PROFILE, seed, ordinal], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & (2**63 - 1)


class TeacherInspection:
    """Named worker RPC; no serialized Python callable is enabled."""

    def xqgeneral_teacher_identity(self):
        from collections import Counter
        model = self.model_runner.model
        counts, methods = Counter(), set()
        for parameter in model.parameters():
            counts[str(parameter.dtype)] += parameter.numel()
        for module in model.modules():
            if getattr(module, 'quant_method', None) is not None:
                methods.add(type(module.quant_method).__name__)
        return {'parameter_elements_by_dtype': dict(counts), 'quant_methods': sorted(methods),
                'model_class': type(model).__name__,
                'model_quant_config_is_none': getattr(model, 'quant_config', None) is None,
                'vision_tower_present': getattr(model, 'visual', None) is not None}


class VllmConsolidator(LocalConsolidator):
    """Share the response/budget contract, with independently seeded requests."""

    def __init__(self, config):
        self.config = config
        self.prefill_chunk_size = None
        if config.get('prefill_chunk_size') is not None:
            raise ValueError('HF prefill chunk settings cannot be applied to the vLLM backend')
        self.generation_profile = (generation_profile(config['generation_profile'])
                                   if config.get('generation_profile') is not None else None)
        self.max_context_tokens = config.get('max_context_tokens')
        checked_budget(1, self.max_context_tokens)
        if self.max_context_tokens is None or (self.generation_profile is not None and
                self.max_context_tokens != self.generation_profile['max_context_tokens']):
            raise ValueError('vLLM requires an explicit, matching teacher context budget')
        if config.get('quantization') != 'none' or config.get('inference_dtype') != 'bfloat16':
            raise ValueError('The authorized consolidator requires unquantized BF16 inference')
        root = Path(config['model_path'])
        self.guard = Artifacts()
        proof = self.guard.json(root / 'weights.manifest.json')
        if (any(proof.get(key) != config[key] for key in ['repository', 'revision', 'quantization']) or
                proof.get('all_official_lfs_sha256_matched') is not True or not proof.get('shards') or
                set(proof.get('stored_tensor_dtypes', {})) != {'BF16'} or
                sum(item['bytes'] for item in proof['shards'].values()) != proof.get('total_weight_bytes')):
            raise ValueError('Full teacher weights require the pinned official BF16 hash manifest')
        for name, item in proof['shards'].items():
            if Path(name).name != name or not name.endswith('.safetensors'):
                raise ValueError('Invalid declared teacher shard')
            self.guard.add(root / name, item)
        for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
            path = root / name
            self.guard.add(path, {'sha256': digest(path), 'bytes': path.stat().st_size})
        model_config = json.loads((root / 'config.json').read_text())
        capacity = model_config.get('text_config', model_config).get('max_position_embeddings')
        if type(capacity) is not int or capacity < self.max_context_tokens:
            raise ValueError('Teacher context exceeds the pinned model capacity')
        self.guard.verify()
        from transformers import AutoTokenizer
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise ImportError('Install the optional teacher-vllm extra in the teacher environment') from error
        self.sampling_class = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = LLM(model=str(root), tokenizer=str(root), dtype='bfloat16', quantization=None,
            tensor_parallel_size=1, max_model_len=self.max_context_tokens, max_num_seqs=MAX_SEQUENCES,
            max_num_batched_tokens=1024, enable_chunked_prefill=True, enable_prefix_caching=True,
            gpu_memory_utilization=0.88, limit_mm_per_prompt={'image': 1, 'video': 0},
            disable_log_stats=True, worker_extension_cls='xqgeneral.vllm_teacher.TeacherInspection')
        try:
            identities = self.model.collective_rpc('xqgeneral_teacher_identity')
            if len(identities) != 1:
                raise ValueError('The teacher backend requires one complete local worker')
            loaded = identities[0]
            dtypes = loaded['parameter_elements_by_dtype']
            if (not dtypes.get('torch.bfloat16') or set(dtypes) - {'torch.bfloat16', 'torch.float32'} or
                    any(type(n) is not int or n <= 0 for n in dtypes.values()) or
                    loaded.get('model_quant_config_is_none') is not True or
                    any(not name.startswith('Unquantized') for name in loaded['quant_methods']) or
                    self.model.llm_engine.vllm_config.model_config.quantization is not None or
                    'vision_config' in model_config and loaded.get('vision_tower_present') is not True):
                raise ValueError('Loaded teacher differs from the complete unquantized BF16 contract')
            self.guard.unchanged()
            self.identity = {'repository': config['repository'], 'revision': config['revision'],
                'quantization': 'none', 'inference_dtype': 'bfloat16', 'backend': 'vllm',
                'parameter_elements_by_dtype': dtypes, 'loaded_worker_identity': loaded,
                'weight_manifest_sha256': digest(root / 'weights.manifest.json'),
                'max_context_tokens': self.max_context_tokens, 'model_context_capacity': capacity,
                'max_num_seqs': MAX_SEQUENCES, 'max_num_batched_tokens': 1024,
                'gpu_memory_utilization': 0.88, 'request_seed_profile': REQUEST_SEED_PROFILE,
                'generation_software': {name: version(name) for name in ['vllm', 'torch', 'transformers']}}
            if self.generation_profile is not None:
                self.identity['generation_profile'] = self.generation_profile
        except BaseException:
            self.model.llm_engine.engine_core.shutdown(timeout=20)
            raise

    def generate(self, messages, max_new_tokens=1024, *, sampling_seed=None):
        return self.generate_batch([messages], max_new_tokens, sampling_seed=sampling_seed)[0]

    def generate_batch(self, messages_batch, max_new_tokens=1024, *, sampling_seed=None):
        self.check_budget(max_new_tokens)
        if not messages_batch:
            return []
        if len(messages_batch) > MAX_SEQUENCES:
            raise ValueError('The vLLM teacher batch exceeds its declared 16-request capacity')
        if self.generation_profile is not None:
            checked_seed(sampling_seed)
        elif sampling_seed is not None:
            raise ValueError('An explicit teacher seed requires a stochastic generation profile')
        prompts, parameters, metadata = [], [], []
        for ordinal, messages in enumerate(messages_batch):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, **self.chat_options())
            tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
            width = len(tokens)
            if width <= 0:
                raise ValueError('Teacher prompt must contain tokens')
            budget = checked_prompt_width(width, max_new_tokens, self.max_context_tokens)
            seed = request_seed(sampling_seed, ordinal) if self.generation_profile is not None else None
            profile = self.generation_profile
            options = ({'temperature': profile['temperature'], 'top_p': profile['top_p'],
                        'top_k': profile['top_k'] or -1} if profile else {'temperature': 0.0})
            prompts.append({'prompt_token_ids': tokens})
            parameters.append(self.sampling_class(max_tokens=budget, seed=seed, **options))
            metadata.append((width, budget, seed))
        self.guard.unchanged()
        outputs = self.model.generate(prompts, parameters, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError('Teacher batch coverage differs from its queries')
        results = []
        for prompt, output, (width, budget, seed) in zip(prompts, outputs, metadata, strict=True):
            if list(output.prompt_token_ids) != prompt['prompt_token_ids'] or len(output.outputs) != 1:
                raise RuntimeError('Teacher generation changed the input prefix or response multiplicity')
            generated = output.outputs[0]
            if not 0 <= len(generated.token_ids) <= budget:
                raise RuntimeError('Teacher generation exceeded its bounded output budget')
            result = self.response(generated.token_ids, width, width, budget, max_new_tokens)
            result.update(finish_reason=generated.finish_reason, generated_token_ids=list(generated.token_ids))
            if seed is not None:
                result['request_sampling_seed'] = seed
            results.append(result)
        self.guard.unchanged()
        return results

    def close(self):
        try:
            self.guard.unchanged()
        finally:
            self.model.llm_engine.engine_core.shutdown(timeout=20)
