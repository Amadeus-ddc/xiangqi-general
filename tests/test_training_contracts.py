import random
import pytest
import torch
from xqgeneral.training import batch_inputs, compatible_resume, sample_batch


class Tokenizer:
    eos_token = '!'
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return 'P'

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def test_answer_only_labels_and_padding():
    rows = [{'id': 'a', 'fen': '9/9/9/9/9/9/9/9/9/9 w - - 0 1', 'question': 'q', 'answer': 'aa'},
            {'id': 'b', 'fen': '9/9/9/9/9/9/9/9/9/9 b - - 0 1', 'question': 'q', 'answer': 'b'}]
    value = batch_inputs(Tokenizer(), rows, 10, 'cpu')
    assert value['labels'].tolist() == [[-100, 97, 97, 33], [-100, 98, 33, -100]]
    assert value['attention_mask'].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    with pytest.raises(ValueError, match='truncated'):
        batch_inputs(Tokenizer(), rows, 2, 'cpu')


def test_ddp_sampling_matches_single_gpu_exposure_and_rng_resume():
    rows = [{'id': str(i), 'stage': s, 'task_type': str(i % 2)} for i in range(10) for s in ['old', 'new']]
    mix = {'old': 0.1, 'new': 0.9}
    one = sample_batch(rows, mix, random.Random(5), 8)
    left = sample_batch(rows, mix, random.Random(5), 8, rank=0, world=2)
    right = sample_batch(rows, mix, random.Random(5), 8, rank=1, world=2)
    assert one == left + right
    rng = random.Random(5)
    sample_batch(rows, mix, rng, 8)
    state = rng.getstate()
    expected = sample_batch(rows, mix, rng, 8)
    restored = random.Random(); restored.setstate(state)
    assert sample_batch(rows, mix, restored, 8) == expected
    with pytest.raises(ValueError, match='divisible'):
        sample_batch(rows, mix, rng, 7, world=2)


def test_resume_rejects_optimizer_or_data_contract_changes():
    config = {'batch_size': 4, 'seed': 1, 'data_path': 'original'}
    compatible_resume(config, dict(config))
    with pytest.raises(ValueError, match='Resume'):
        compatible_resume(config, dict(config, data_path='changed'))
