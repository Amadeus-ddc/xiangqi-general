# Single-GPU FP32 Adam state offload

Set `optimizer_state_offload` to `"cpu"` and `optimizer_foreach` to `false`
for single-rank, unsharded training. Existing configurations retain their
ordinary AdamW behavior. Changing this setting requires a new experiment;
exact resume refuses a change of memory configuration.

Parameters and gradients remain FP32 on the model device. Between optimizer
updates, Adam's first and second moments reside on CPU. During the optimizer
step, one parameter's moments move to its GPU, native PyTorch single-tensor
AdamW updates that parameter, and its moments return to CPU. This reduces
persistent GPU storage without quantization, parameter freezing or a change
to the learning rate, gradient accumulation, clipping or AdamW arithmetic.
The host must hold the full moments; copies add step time. DDP, FSDP, ZeRO,
foreach/fused/capturable/differentiable optimizers are unsupported with this
setting.

Checkpoints keep native AdamW parameter groups and state identifiers. Loading
uses CPU proxy parameters to prevent a transient full-GPU moment allocation.
Each saved training checkpoint has an `optimizer-placement-step-N.json`
companion with actual state placement and largest active moment storage.

The motivating full-model GPU0 probes both completed one temporary update
and failed during the second backward. The default allocator reported
70.48 GiB allocated and 2.05 GiB unused reserved storage; expandable segments
reported 71.47 GiB allocated, about 0.45 GiB unused reserved and 1.29 GiB free
for a 1.45 GiB request. Other occupation was about 21.35 GiB in both errors.
These failures are preserved and the selected model is unchanged.

CPU tests compare actual multi-group AdamW updates and exact interrupted
continuation, including parameter ownership, missing gradients and malformed
checkpoints. CPU tests cannot prove CUDA capacity or GPU arithmetic parity.
The unchanged full-model continuous-four-versus-two-plus-two probe must still
pass with all 624 trainable tensors and FP32 moments before formal training.
No successful full-model offload probe or search-SGD round is claimed here.
