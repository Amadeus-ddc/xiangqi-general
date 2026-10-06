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


def load_model(config, expert_dim=512, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
    base = AutoModelForCausalLM.from_pretrained(config['model_path'], local_files_only=True,
                                               dtype=dtype, attn_implementation='sdpa').to(device)
    mode = config.get('mode', 'bridge')
    if mode == 'bridge':
        model = BoardLanguageModel(base, expert_dim, config['decoder_bridge_positions'], config['bridge_width'])
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
    return model.to(device=device, dtype=dtype), tokenizer


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


def load_checkpoint(path, device='cuda'):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model, tokenizer = load_model(checkpoint['config'], device=device)
    load_trainable(model, checkpoint)
    model.eval()
    return model, tokenizer, checkpoint['config']
