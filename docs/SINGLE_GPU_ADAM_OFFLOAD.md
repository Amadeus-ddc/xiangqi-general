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
The host must hold the full moments; copies add step time. Multi-rank DDP, FSDP, ZeRO,
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

The explicit search profile is
`full_decoder_search_one_worker_cpu_moments_v1`, supplied by
`configs/search-distillation-sft-one-gpu-cpu-offload-with-auxiliary-v1.json`.
It keeps global batch 30, microbatch 1, full FP32 parameters, token-normalized
loss, finite coverage of every new label and all six replay tasks, and complete
seven-stage validation. The recipe uses the existing single-rank `ddp`
configuration value; no distributed wrapper is constructed at world size 1.
Other search profiles cannot silently opt into CPU moment offload.

After native data acceptance, immutable feature extension, token preflight and
the required original development baseline are complete, run in a new directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m xqgeneral.distillation_training \
  --init /path/to/selected-parent/adapter.pt \
  --recipe configs/search-distillation-sft-one-gpu-cpu-offload-with-auxiliary-v1.json \
  --token-preflight /path/to/new-data-token-preflight/manifest.json \
  --output runs/new-one-worker-search-round
```

Bind `replay_data_paths` to the actual accepted replay dataset before launch.
Use `--resume` only for the same immutable experiment. Physical GPU allocation
must follow the user's current reservation; this example does not grant access
to another card. These commands load the real model and execute the real GPU
preflight before formal updates; a recipe alone does not establish capacity.

The native preflight requires `optimizer-placement-step-2.json` and
`optimizer-placement-step-4.json` from both continuous and resumed execution.
Each must bind its actual update and describe all trainable tensors, three
state tensors per parameter, FP32 CPU moments, CUDA parameter placement and
the exact complete state byte count. Formal completion and later parent
acceptance independently reject missing or incompatible runtime residency.
CPU integration controls replace only the GPU producers with explicitly
constructed fixtures, then exercise native preflight, training orchestration,
completion reopening, resume, and normal/zero/shuffled raw validation. They
do not prove a real single-GPU search round or coaching gain.
