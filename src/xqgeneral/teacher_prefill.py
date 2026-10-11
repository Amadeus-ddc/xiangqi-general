"""Bound transient teacher prefill tensors while preserving the whole prompt."""
import torch


def checked_prefill_chunk_size(value):
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError('Teacher prefill chunk size must be a positive integer')
    return value


@torch.inference_mode()
def generate_prefilled(model, inputs, chunk_size=None, **options):
    chunk_size = checked_prefill_chunk_size(chunk_size)
    if chunk_size is None:
        return model.generate(**inputs, **options)
    if set(inputs) != {'input_ids', 'attention_mask'}:
        raise ValueError('Chunked teacher prefill requires token IDs and a text attention mask')
    ids, mask = inputs['input_ids'], inputs['attention_mask']
    if (ids.ndim != 2 or mask.shape != ids.shape or not ids.shape[0] or not ids.shape[1] or
            not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask[:, -1] == 1) or
            torch.any(mask[:, 1:] < mask[:, :-1])):
        raise ValueError('Chunked teacher prefill requires nonempty text with contiguous left padding')
    # Cached linear attention no longer masks padding. Finish every row's padding
    # in the first chunk, with at least one real token to establish its RoPE offset.
    if torch.any((mask == 0).sum(1) >= chunk_size):
        raise ValueError('Teacher prompt lengths must be grouped before chunked prefill')
    if options.get('use_cache') is False or 'past_key_values' in options:
        raise ValueError('Chunked teacher prefill requires its own fresh enabled cache')
    width = ids.shape[1]
    if width <= chunk_size:
        return model.generate(**inputs, **options)
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
    cache = None
    # Let generate() process the last complete input chunk with its normal
    # prefill path, then return the whole original input followed by new tokens.
    prefix_width = ((width - 1) // chunk_size) * chunk_size
    for start in range(0, prefix_width, chunk_size):
        end = start + chunk_size
        output = model(input_ids=ids[:, start:end], attention_mask=mask[:, :end],
                       position_ids=positions[:, start:end], past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        cache = getattr(output, 'past_key_values', None)
        if cache is None or not callable(getattr(cache, 'get_seq_length', None)) or cache.get_seq_length() != end:
            raise RuntimeError('Teacher prefill cache does not cover the exact complete input prefix')
        del output
    return model.generate(**inputs, **{**options, 'position_ids': positions,
                                     'use_cache': True, 'past_key_values': cache})
