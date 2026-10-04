# AdamW training

Select `nnUNetTrainerAdamW` using the existing training or batch-training command.
The standard environment (`uv sync --extra dev --locked`) includes its CPU
augmentation dependencies. No external trainer module or optional extra is needed.

The trainer uses reusable shared CPU batch buffers when augmentation workers feed
CUDA training with pinning enabled. Windows workers use spawn; Linux workers use
the active multiprocessing context. Returned batches own their pinned tensors,
so a worker cannot overwrite a retained batch or an in-flight GPU transfer.
Linux containers with insufficient `/dev/shm` capacity fall back to the ordinary
augmentation queue and emit a warning. `nnUNet_n_proc_DA` still sets the number of
augmentation workers per training job. Zero selects single-process preparation.

Gaussian blur and region conversion use accelerated CPU implementations. Supported
3D spatial label interpolation uses a fused CPU kernel, preserving the original
grid, image interpolation, score rounding, label ties and constant-padding rule.
Unsupported label interpolation modes and 2D/dummy-2D spatial operations retain
the original transform. The kernel compiles on first use and caches its compilation.

The configuration's `trainer` dictionary retains these parameters:

- `initial_lr`, `weight_decay`, `num_epochs`, `warmup_epochs`, `min_lr`
- `enable_deep_supervision`
- `router_schedule` with `start_val`, `end_val`, `end_epoch`
- `pin_memory`

An explicit `trainer.pin_memory` value takes precedence over `nnUNet_pin_memory`.
With pinning disabled, the ordinary queue is used and the CPU transform
optimizations remain active. Existing device-dependent pinning defaults are retained.

Training and batch-training keep their existing configuration selection, fold
ordering, checkpoint/resume, `--disable-tta`, `--disable_train_val`, and
`--ckpt-interval` behavior. Existing AdamW checkpoints retain their trainer name
and constructor arguments. The optimizer and learning-rate/router schedules are
unchanged.

Regression checks are part of `uv run --with pytest pytest nnunetv2/tests`.
The numerical and shared-buffer lifecycle checks run on both platforms; their
pinning/worker checks require a visible CUDA device. Run the spatial comparisons
when upgrading PyTorch or batchgeneratorsv2, since exact resampling behavior is
part of the trainer's compatibility contract.
