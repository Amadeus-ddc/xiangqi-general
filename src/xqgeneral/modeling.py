"""Model-specific loading and checkpoint contracts shared by training/inference."""
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from .bridge import BoardLanguageModel, GatedBridge


class TextLanguageModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    @contextmanager
    def board_context(self, features):
        yield

    def forward(self, **kwargs):
        return self.base(**kwargs)


def configure_trainable_precision(model, precision):
    if precision not in {'base', 'float32'}:
        raise ValueError('Trainable precision must be base or float32')
    if precision == 'float32':
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()


def load_model(config, expert_dim=512, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Full SFT keeps FP32 master parameters so small decoder updates are retained.
    dtype = (torch.bfloat16 if str(device).startswith('cuda') and
             config.get('decoder_training', 'frozen') != 'full' else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(config['model_path'], local_files_only=True,
                                               dtype=dtype, attn_implementation='sdpa').to(device)
    mode = config.get('mode', 'bridge')
    decoder_training = config.get('decoder_training', 'frozen')
    if decoder_training not in {'frozen', 'full'}:
        raise ValueError('Decoder training must be explicitly frozen or full')
    if mode == 'bridge':
        model = BoardLanguageModel(base, expert_dim, config['decoder_bridge_positions'], config['bridge_width'],
                                   freeze_decoder=decoder_training == 'frozen')
    elif mode == 'text_lora':
        from peft import LoraConfig, get_peft_model
        base = get_peft_model(base, LoraConfig(r=config.get('lora_rank', 16), lora_alpha=config.get('lora_rank', 16) * 2,
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                             lora_dropout=0.0, task_type='CAUSAL_LM'))
        model = TextLanguageModel(base)
    else:
        raise ValueError(f"Unsupported model mode {mode}")
    if config.get('board_tokens', False):
        from .board_tokens import install_board_tokens
        install_board_tokens(base, tokenizer)
    if decoder_training == 'full':
        base.requires_grad_(True)
    model.to(device=device, dtype=dtype)
    configure_trainable_precision(model, config.get('trainable_parameter_dtype', 'base'))
    return model, tokenizer


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def foundation_state_summary(checkpoint, base_config):
    """Check the complete frozen-decoder bridge/token state on CPU without base weights."""
    from .board_tokens import PAIRS
    config = checkpoint['config']
    positions = config['decoder_bridge_positions']
    if (config.get('mode') != 'bridge' or config.get('decoder_training') != 'frozen' or
            config.get('board_text') is not None or config.get('board_tokens') is not True or
            config.get('trainable_parameter_dtype') != 'float32' or
            not positions or len(set(positions)) != len(positions) or
            len(positions) != len(config['expert_feature_depths']) or
            any(type(p) is not int or not 0 <= p < base_config['num_hidden_layers'] for p in positions)):
        raise ValueError('Handoff requires the complete latent FP32 foundation architecture')
    hidden = base_config['hidden_size']
    with torch.device('meta'):
        bridge = GatedBridge(hidden, 512, config['bridge_width'])
    expected = {f'bridges.{i}.{name}': parameter.shape
                for i in range(len(positions)) for name, parameter in bridge.named_parameters()}
    expected.update({name: (len(PAIRS), hidden) for name in
        ['base.model.embed_tokens.board_weight', 'base.lm_head.board_weight']})
    state = checkpoint['trainable']
    if set(state) != set(expected):
        raise ValueError('Foundation checkpoint must contain every bridge and both board embeddings')
    if any(not isinstance(t, torch.Tensor) or t.shape != expected[name] or
           t.dtype != torch.float32 or t.device.type != 'cpu' or not torch.isfinite(t).all()
           for name, t in state.items()):
        raise ValueError('Foundation checkpoint tensors have incompatible shapes, precision or values')
    return {'trainable_tensors': len(state), 'trainable_parameters': sum(t.numel() for t in state.values()),
            'all_bridge_and_board_shapes_checked': True, 'all_trainable_values_finite_fp32': True,
            'base_model_weights_loaded': False}


def load_trainable(model, checkpoint):
    if 'trainable' in checkpoint:
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        state = checkpoint['trainable']
        if set(state) != expected:
            raise ValueError("Checkpoint trainable parameter contract differs from model configuration")
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                if parameters[name].shape != value.shape:
                    raise ValueError(f"Checkpoint shape mismatch at {name}")
                parameters[name].copy_(value)
    elif 'bridges' in checkpoint and hasattr(model, 'bridges'):
        model.bridges.load_state_dict(checkpoint['bridges'])
    else:
        raise ValueError("Unsupported checkpoint format")


def initialize_trainable(model, checkpoint, config):
    saved = checkpoint['config']
    architecture = ['mode', 'decoder_bridge_positions', 'bridge_width', 'board_tokens', 'model_revision', 'board_text']
    if any(saved.get(k) != config.get(k) for k in architecture):
        raise ValueError('Initialization architecture differs from the selected checkpoint')
    expanding = saved.get('decoder_training', 'frozen') == 'frozen' and config.get('decoder_training') == 'full'
    if not expanding:
        load_trainable(model, checkpoint)
        return
    parameters = dict(model.named_parameters())
    state = checkpoint['trainable']
    if not state or any(n not in parameters or not parameters[n].requires_grad or
                        parameters[n].shape != v.shape for n, v in state.items()):
        raise ValueError('Expanded SFT initialization has an incompatible parameter contract')
    required = {n for n, p in parameters.items() if p.requires_grad and
                (n.startswith('bridges.') or n.endswith('board_weight') or 'lora_' in n)}
    if set(state) != required:
        raise ValueError('Expanded SFT initialization must load all bridges and board embeddings')
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def expanded_initialization_reference(config, source_path):
    """Build the complete CPU initial state for a guarded full-decoder probe."""
    if config.get('decoder_training') != 'full' or config.get('trainable_parameter_dtype') != 'float32':
        raise ValueError('The full-decoder reference requires FP32 trainable parameters')
    source = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    model, _ = load_model(config, device='cpu')
    initialize_trainable(model, source, config)
    state = trainable_state(model)
    if (len(state) <= len(source['trainable']) or
            any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()) or
            any(name not in state or not torch.equal(t, state[name]) for name, t in source['trainable'].items())):
        raise ValueError('Expanded decoder reference must preserve every foundation tensor exactly')
    return state


def load_checkpoint(path, device='cuda'):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model, tokenizer = load_model(checkpoint['config'], device=device)
    load_trainable(model, checkpoint)
    model.eval()
    return model, tokenizer, checkpoint['config']
