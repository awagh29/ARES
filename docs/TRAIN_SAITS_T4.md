# Train SAITS endpoint reconstruction on a Colab T4

This guide trains the project's supervised SAITS adaptation on the existing single-episode preparations. The trainer takes `train_prepared` and `val_prepared`, fits preprocessing using training rows only, selects a checkpoint using validation, and saves the state needed to resume. Test evaluation is a separate call after training.

**Interpret this episode as a pipeline check.** In the existing first 302-scan flight, all seven forward sectors have no returns throughout validation and test. The 14 clean targets there are constant: seven distances of 7 m and seven return fractions of 0. Persistence already achieves zero reconstruction error on these targets. Training curves and held-out errors can verify the implementation, but cannot establish obstacle reconstruction skill or performance on new flights. The synthetic experiment also retains the original healthy-flight motion; it does not simulate navigation with an actual sensor failure.

## 1. Use the existing Colab environment

Select a GPU runtime in Colab, then inspect the runtime you actually received:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Select a GPU runtime before starting training."
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
```

Keep Colab's installed PyTorch when CUDA works. The trainer needs NumPy, pandas, and PyTorch; matplotlib is optional for plotting. If the other packages are missing, install them in a separate cell:

```python
%pip install numpy pandas matplotlib
```

Use the existing ARES checkout and data. There is no need to clone the original SAITS repository or install PyPOTS for this project implementation.

```python
from pathlib import Path
import sys

repo_root = Path("/content/ares")  # Change to your existing checkout.
assert (repo_root / "scripts" / "train_saits_t4.py").is_file()
sys.path.insert(0, str(repo_root / "scripts"))

from train_saits_t4 import TrainConfig, train_saits, evaluate_saits, plot_training
```

## 2. Keep the prepared splits and fault schedules fixed

Use `train_prepared`, `val_prepared`, and `test_prepared` from [the single-episode preparation guide](SINGLE_EPISODE_SAITS.md). If those variables already exist, retain them. Do not redraw outages between epochs or regenerate the test schedule while choosing a model.

For a fresh preparation from the existing clean `df`:

```python
from lidar_reconstruction_data import (
    split_train_val_test,
    introduce_lidar_perturbations,
    make_saits_windows,
)

train_df, val_df, test_df = split_train_val_test(df)

train_prepared = introduce_lidar_perturbations(
    train_df, seed=42, n_outages=3, window_size=50,
)
val_prepared = introduce_lidar_perturbations(
    val_df, seed=43, n_outages=1, window_size=50,
)
test_prepared = introduce_lidar_perturbations(
    test_df, seed=44, n_outages=1, window_size=50,
)

train_windows = make_saits_windows(train_prepared)
val_windows = make_saits_windows(val_prepared)
test_windows = make_saits_windows(test_prepared)

for name, prepared, windows in [
    ("train", train_prepared, train_windows),
    ("validation", val_prepared, val_windows),
    ("test", test_prepared, test_windows),
]:
    print(name, windows["X"].shape, windows["y"].shape,
          prepared["metadata"]["scored_target_content"])
```

The 302-row episode splits into 181/60/61 rows. With the current default schedules, the scored endpoint windows number 68/6/8. Each input window has shape `[50,31]`; each label has 14 features. These overlapping windows are dependent observations from one flight, not independent trials.

The trainer receives the prepared dictionaries, not just `train_windows`: it needs the unique training rows to fit context normalization without counting overlapping history repeatedly. It builds the same past-to-present windows internally. All history, outage age, position increments, and persistence state remain inside their own split.

## 3. Train and save the selected model

Choose a new output directory for each experiment. For persistent Colab storage, use a Drive location only after you have mounted Drive yourself:

```python
drive_root = Path("/content/drive/MyDrive")
assert drive_root.is_dir(), "Mount your Drive in Colab, or choose a local output directory."
output_dir = drive_root / "ARES" / "saits" / "episode_001_t4_seed42"

config = TrainConfig(
    device="cuda",
    amp=True,
    epochs=200,
    batch_size=128,
    d_model=64,
    n_heads=4,
    d_ff=128,
    n_layers=1,
    dropout=0.1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=25,
    min_delta=1e-5,
    checkpoint_every=10,
    log_every=10,
)

result = train_saits(
    train_prepared,
    val_prepared,
    output_dir=output_dir,
    config=config,
)
print("Device:", result.device)
print("Best epoch:", result.best_epoch)
print("Best validation metric:", result.best_metric)
display(result.history.tail())
```

The training API deliberately has no test argument. Validation guides checkpoint selection and early stopping; test labels do not participate in training, preprocessing, learning-rate decisions, or epoch selection. The returned `TrainingResult` exposes `model`, `transform`, `history`, `best_epoch`, `best_metric`, `output_dir`, and `device`.

Optional plotting uses the saved history and does not run another evaluation:

```python
fig = plot_training(result)
display(fig)
# Optional: fig.savefig(output_dir / "training_curves.png", dpi=160, bbox_inches="tight")
```

## 4. Evaluate the fixed test split after selection

Run this after training and model selection are complete:

```python
test_metrics = evaluate_saits(result, test_prepared, split_name="test")
print(test_metrics)
```

Evaluation uses the selected model and the training-fitted transformation. It saves metrics and endpoint predictions, including persistence predictions, for the same scored timestamps. Distance error is reported in metres separately from return-fraction error; a combined scaled training loss is not an error in metres. Keep the counts and target-content diagnostic with the reported scores.

For the current first episode, `no-return-only` validation/test targets and a perfect persistence baseline make this a plumbing result. More epochs or a larger network do not remedy that lack of target variation. Preserve the chosen split and report its limitation explicitly.

## Model, preprocessing, and loss

The model uses two diagonally masked self-attention blocks and a learned combination gate, with independent layer parameters. This is a supervised endpoint adaptation of the [SAITS architecture](https://github.com/WenjieDu/SAITS), rather than the stock library training objective. The primary reconstruction loss is masked Huber loss on the 14 missing features at the final row of each window, with auxiliary supervision of the intermediate reconstructions weighted by 0.1. The clean labels stay separate from model inputs.

Each input contains 14 LiDAR sector features followed by 17 motion and availability features. The transform divides the first seven distance channels by the physical maximum range of 7 m and leaves the seven fractions unchanged. It fits the 17 context channels on unique chronological training rows, handling constant channels without division by zero. Validation and test reuse those statistics. The same target transformation applies to labels and persistence, and evaluation converts distance predictions back to metres.

Missing inputs retain their observation mask. A finite placeholder used for tensor operations is never marked as an observed reading. Healthy sectors without valid returns remain observed `[7 m,0]`; a synthetic outage makes the sector inputs unavailable. Controller actions, avoidance references, and status codes remain outside the predictor set.

A window ends at the target time. Attention can use its historical rows, but reconstruction is supervised and scored only at the final row. Earlier rows are not scored as if future rows within the window were unavailable.

## T4 execution choices

The small datasets are placed on the selected device once as float32 tensors. CUDA execution uses FP16 automatic mixed precision and gradient scaling where appropriate; numerically sensitive final attention calculations and loss reduction use float32. PyTorch's [AMP guidance](https://docs.pytorch.org/docs/stable/amp.html) explains the roles of autocasting and gradient scaling.

Other eligible attention operations use scaled dot-product attention with automatic backend selection. The trainer does not force a FlashAttention backend that may be unavailable on the runtime GPU. See [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for backend selection and numerical differences.

The [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) is a Turing GPU with FP16 Tensor Cores and 16 GB of memory. These settings use FP16 rather than assuming native BF16 or TF32 support. A compatible fused CUDA AdamW implementation may be used when available, with a standard optimizer fallback. The tiny example usually fits into a single training batch at the default batch size; high sustained GPU utilization is not an appropriate success criterion for 68 training windows.

No T4 timing claim follows from a local CPU test. Inspect the actual Colab device, elapsed times, selected configuration, and saved history before describing performance.

Local verification passed 31 Python tests, including gradients through both stages and the combination gate, diagonal attention masking, training-only normalization, and exact CPU checkpoint recovery with dropout. A five-epoch CPU smoke run also completed on the actual first episode with the default model dimensions and 68/6/8 train/validation/test windows, producing all checkpoint and evaluation files. CUDA execution and T4 speed have not been measured locally.

## Saved files and resuming

The output directory contains:

- `run_config.json`: experiment, source fingerprints, runtime settings, and resume-session history.
- `preprocessing.json`: fitted feature transformations.
- `history.csv`: epoch-level training and validation history.
- `training_summary.json`: the selected epoch, validation result, and run summary.
- `best.pt`: the selected model checkpoint.
- `last.pt`: the latest saved resumable training checkpoint.
- `validation_metrics.json` and `validation_predictions.npz`/`.csv`: validation outputs.
- `test_metrics.json` and `test_predictions.npz`/`.csv`: outputs created by the separate test-evaluation call.

Checkpoints contain model, optimizer, scheduler, gradient-scaler, random-generator, preprocessing, configuration, history, and data-fingerprint state. Writes use temporary files and replacement, so a complete previous checkpoint survives an interrupted write. Periodic saving avoids writing Drive files every minibatch; interruption can still lose work since the most recent completed checkpoint.

To continue the same experiment, recreate the same prepared partitions and configuration, then use its existing output directory:

```python
from dataclasses import replace

resumed_config = replace(config, epochs=300)  # Extending the epoch limit is allowed.
result = train_saits(
    train_prepared,
    val_prepared,
    output_dir=output_dir,
    config=resumed_config,
    resume=True,
)
```

Resume checks the data fingerprints (including supervision masks and persistence), preprocessing, source files, and training configuration. Keep the same rows, feature order, fault schedules, architecture, and optimization settings. A new split, outage seed, or hyperparameter experiment belongs in a new directory. The trainer resumes from the latest saved state and retains the selected checkpoint; it does not substitute the test score for validation. Exact numerical replay across different PyTorch/CUDA versions or GPU kernels is not guaranteed. Prediction exports include the episode run ID and scored timestamps.

If the GPU runs out of memory, use a new experiment directory with a smaller `batch_size` such as 32. If mixed-precision operations produce nonfinite values, inspect the reported failure and use a new run with `amp=False` to diagnose it. Do not change precision or model settings in the middle of a resumable experiment while claiming it is the same configuration.
