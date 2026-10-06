"""Reusable, oracle-free model inference with explicit history contracts."""
import torch
from .evidence import history_key
from .expert import FrozenPx0, encode_history
from .modeling import load_checkpoint
from .rules import replay
from .training import messages
from .board_tokens import decode_board_text, encode_board_text


class Predictor:
    def __init__(self, checkpoint, weights='models/px0-latest.pb.gz', device='cuda', feature_cache=None):
        torch.set_num_threads(4)
        self.device = device
        self.model, self.tokenizer, self.config = load_checkpoint(checkpoint, device)
        self.cache = torch.load(feature_cache, map_location='cpu', weights_only=True) if feature_cache else None
        self.indices = {k: i for i, k in enumerate(self.cache['keys'])} if self.cache else {}
        self.expert = FrozenPx0(weights).to(device) if self.config.get('mode', 'bridge') == 'bridge' else None

    @torch.no_grad()
    def features(self, record):
        if self.expert is None:
            return []
        history = record.get('history')
        if history is None:
            history = replay(record['initial_fen'], record['moves'])
        if history[-1] != record['fen']:
            raise ValueError('Inference board differs from the final history position')
        key = history_key(history)
        if key in self.indices:
            return [f[self.indices[key]:self.indices[key] + 1].to(self.device, torch.bfloat16)
                    for f in self.cache['features']]
        depths = self.config.get('expert_feature_depths')
        kwargs = {'depths': depths} if depths is not None else {}
        features, _ = self.expert(encode_history(history).unsqueeze(0).to(self.device), **kwargs)
        return [f.to(torch.bfloat16) for f in features]

    @torch.no_grad()
    def generate(self, record, question=None, max_new_tokens=768):
        return self.generate_batch([dict(record, question=question or record['question'])],
                                   max_new_tokens=max_new_tokens)[0]

    @torch.no_grad()
    def generate_batch(self, records, max_new_tokens=768, memory='normal'):
        if not records:
            return []
        if memory not in {'normal', 'zero', 'shuffled'}:
            raise ValueError('Unknown expert-memory ablation')
        texts = [self.tokenizer.apply_chat_template(messages(r, self.config.get('mode', 'bridge')),
                                                   tokenize=False, add_generation_prompt=True) for r in records]
        if self.config.get('board_tokens', False):
            texts = [encode_board_text(t) for t in texts]
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        try:
            inputs = self.tokenizer(texts, return_tensors='pt', add_special_tokens=False, padding=True).to(self.device)
        finally:
            self.tokenizer.padding_side = previous_padding
        by_record = [self.features(r) for r in records]
        features = [torch.cat([f[level] for f in by_record]) for level in range(len(by_record[0]))]
        if memory == 'zero':
            features = [torch.zeros_like(f) for f in features]
        elif memory == 'shuffled':
            if len(records) < 2:
                raise ValueError('Memory shuffling needs at least two histories')
            features = [torch.roll(f, 1, 0) for f in features]
        with self.model.board_context(features):
            outputs = self.model.base.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                               pad_token_id=self.tokenizer.pad_token_id)
        answers = [self.tokenizer.decode(o[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                   for o in outputs]
        return [decode_board_text(a) for a in answers] if self.config.get('board_tokens', False) else answers
