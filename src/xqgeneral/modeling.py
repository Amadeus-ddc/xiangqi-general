"""Model-specific loading and checkpoint contracts shared by training/inference."""
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from .bridge import BoardLanguageModel


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


def load_checkpoint(path, device='cuda'):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model, tokenizer = load_model(checkpoint['config'], device=device)
    load_trainable(model, checkpoint)
    model.eval()
    return model, tokenizer, checkpoint['config']
