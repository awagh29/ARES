# SAITS experiment with one LiDAR episode

This workflow uses the existing 302-row episode for a small, reproducible reconstruction experiment. It splits the flight chronologically, introduces a fixed LiDAR outage schedule separately within each partition, and builds past-to-present windows for SAITS. The implementation is in [`scripts/lidar_reconstruction_data.py`](../scripts/lidar_reconstruction_data.py).

The held-out result measures reconstruction on later portions of this one healthy flight after synthetic masking. It does not establish performance on new trajectories or navigation performance while the simulator actually loses LiDAR. This experiment has train, validation, and test partitions; it does not add a calibration partition or claim a safety guarantee.

**Observed limitation of the specific first episode:** all seven forward sectors have no valid returns throughout both the validation and test partitions. Their clean targets are constant: seven distances of 7 m and seven fractions of 0. Persistence therefore already gives zero error on those scored targets. This split is useful for checking the implementation, but cannot provide an informative validation/test comparison for obstacle reconstruction. Do not present a small error on these empty-view segments as evidence that the model reconstructs obstacles. An informative reconstruction evaluation needs a single episode or a predeclared segment whose held-out portion actually includes returns.

## Data and prediction target

Each row represents one LiDAR scan. The seven sectors cover the forward body-frame field of view from -60 to +60 degrees. Each sector uses all 15 elevation channels and produces two targets: an obstacle-distance feature and a valid-return fraction. Together these give 14 target features.

Use the actual sensor geometry when assigning beams. The organized scan has 716 sensor azimuth columns spanning `linspace(-179, 179, 716)`. The configured 180-degree sensor roll reverses horizontal bearing, so `body_bearing = -sensor_azimuth`. Grouping sensor columns directly by their signed azimuth would reverse the body-frame sector order.

A range of 7 metres with a false valid-return mask means a normal absent return. It is not a sensor outage. Synthetic outages replace the 14 input sector features with missing values while preserving their original clean values only as training/scoring labels.

The input has 31 features in this order:

1. The 14 current LiDAR sector features, masked during the synthetic outage.
2. Three position-increment features.
3. Three velocity features.
4. Six sine/cosine attitude features for roll, pitch, and yaw.
5. Three body angular-rate features.
6. One known sensor-availability feature.
7. One time-since-last-healthy-scan feature.

The last 17 features provide motion and availability context. Controller actions, obstacle-avoidance references, and status codes are excluded because, in this healthy-flight recording, they can encode the controller's response to the LiDAR that the experiment is trying to reconstruct.

## Chronological split

Call `split_train_val_test(df)` before generating outages, fitting preprocessing, or constructing windows. For 302 rows, the chronological 60/20/20 split contains 181 training rows, 60 validation rows, and 61 test rows.

All window history and outage state stay within their own partition. Availability age, position differences, and persistence predictions must be initialized independently at each boundary. The helper reserves a healthy predecessor and sufficient local history for scoring; it does not borrow the final training scan to initialize validation or the final validation scan to initialize test.

The default synthetic fault profile uses full-scan outages of 1.5-3 seconds,
at least 3 seconds of initial healthy data, and at least 2 seconds of healthy
recovery between outages, matching the relevant MATLAB configuration. Placement
also ensures the requested minimum of scored endpoints after local history.

At a 50-row window length, these partitions contain at most 132, 11, and 12 complete windows respectively. Only windows whose final row is a scored outage target contribute to the reconstruction loss or outage metric. Validation and test therefore have very few usable targets. Each generated schedule reserves at least five scored missing endpoints, but those overlapping windows are strongly dependent and do not constitute five independent flight trials.

## Prepare the experiment in Colab

Use the existing episode-loading cell and its `episode` or `df` variable. Upload [`scripts/lidar_reconstruction_data.py`](../scripts/lidar_reconstruction_data.py) into the Colab working directory, or add the repository's `scripts` directory to the Python import path. This module contains `make_table`, so an additional notebook implementation is unnecessary.

```python
from lidar_reconstruction_data import (
    make_table,
    split_train_val_test,
    introduce_lidar_perturbations,
    make_saits_windows,
)

# Keep an existing df, or construct it from the already loaded episode.
if "df" not in globals():
    df = make_table(episode)

train_df, val_df, test_df = split_train_val_test(df)
assert (len(train_df), len(val_df), len(test_df)) == (181, 60, 61)

# Generate once, then reuse these exact schedules throughout this experiment.
train_prepared = introduce_lidar_perturbations(
    train_df, seed=42, n_outages=3,
    duration_s=(1.5, 3.0), window_size=50, min_scored_steps=5,
)
val_prepared = introduce_lidar_perturbations(
    val_df, seed=43, n_outages=1,
    duration_s=(1.5, 3.0), window_size=50, min_scored_steps=5,
)
test_prepared = introduce_lidar_perturbations(
    test_df, seed=44, n_outages=1,
    duration_s=(1.5, 3.0), window_size=50, min_scored_steps=5,
)

train_windows = make_saits_windows(train_prepared)
val_windows = make_saits_windows(val_prepared)
test_windows = make_saits_windows(test_prepared)

for name, windows in [
    ("train", train_windows),
    ("validation", val_windows),
    ("test", test_windows),
]:
    print(name, windows["X"].shape, windows["y"].shape)
```

Check target coverage before training:

```python
for name, prepared in [("train", train_prepared),
                       ("validation", val_prepared), ("test", test_prepared)]:
    info = prepared["metadata"]
    print(name, info["scored_endpoint_count"], info["scored_target_content"],
          info["scored_endpoints_with_returns"])
```

For this first episode, validation/test report `no-return-only` and zero scored
endpoints with returns. The functions expose this diagnostic without silently
changing the selected episode, split, targets, or test schedule.

The assertion applies to this 302-row episode. For another episode, inspect its split sizes rather than changing the split to reproduce these counts.

Do not regenerate the test schedule to seek a better score. Keep the seeds and `outage_intervals` with the experiment settings. Use validation for model choices, then evaluate the selected model on the fixed test schedule once.

## Helper outputs and masks

`introduce_lidar_perturbations(...)` returns:

- `X`: a `[T, 31]` input array. Its first 14 features contain `NaN` during an outage.
- `y`: the clean `[T, 14]` reconstruction targets.
- `target_mask`: the outage targets, including outage rows without sufficient window history.
- `score_mask`: outage targets with at least 49 preceding rows within the same partition.
- `observed_df`: the corrupted observations, including the corresponding corrupted raw arrays.
- `outage_intervals`: the fixed outage schedule.
- `persistence_y`: predictions formed from the most recent healthy observation available in that partition.

`make_saits_windows(prepared)` returns:

- `X`: `[N, 50, 31]`, retaining missing entries as `NaN`.
- `observed_mask`: boolean `[N, 50, 31]`; true means observed.
- `y`: clean `[N, 14]` labels at the final row of each window.
- `target_mask`: `[N, 14]`, identifying the missing final-row targets to score.
- `target_times_s` and `source_row_positions`: the target timestamps and row positions for joining predictions back to the partition.
- `target_feature_indices`: input feature indices `0` through `13`.

Use the masks for their stated purposes. `observed_mask` is an input-availability mask; it is not a label for which errors belong in the loss. `target_mask` determines the final-row reconstruction labels that contribute to loss and evaluation. The normal LiDAR valid-return mask describes beam returns and must remain distinct from both.

## Preprocessing

Fit any learned preprocessing on training data only. Apply the same fitted transformation to validation and test without refitting. Preserve missing entries as `NaN` while preparing arrays, and preserve the accompanying observation masks.

For distance targets, dividing by the fixed physical maximum range of 7 metres is a consistent alternative to fitting a distance scaler. Valid-return fractions already lie in `[0, 1]`. Whichever target transformation is chosen, apply it consistently to observed input sector features, clean labels, model outputs, and the persistence baseline, and invert it when reporting distance errors in metres.

For the 17 motion/availability context features, derive normalization statistics from the training partition only. Handle constant features without division by zero. Do not fit a scaler to the full flight, validation/test labels, or the concatenation of all windows. Overlapping windows should not silently give repeated training rows extra weight when computing scaling statistics.

## Adapt SAITS to this supervised task

SAITS is an imputation model; this experiment asks it to reconstruct 14 specific features at the last, missing row of a historical window. Its training adapter must implement that objective explicitly. Passing the returned arrays directly into an unchanged stock `fit` method does not automatically train against the held-out clean outage labels.

For a batch:

1. Supply the corrupted `X` and its observation mask to the model. If the implementation requires finite tensors, replace `NaN` with a fixed placeholder only alongside that mask.
2. Select the predictions for the final time step and feature indices `0:14`.
3. Compare those predictions with the separate clean `y` labels, using `target_mask` to include only the missing endpoint targets.
4. Aggregate validation loss with the same masking rule. Select checkpoints and hyperparameters using validation only.

The clean `y` values are labels. Never insert them into the corresponding missing input entries or expose them through preprocessing. A placeholder such as zero is also not an observed zero-distance reading: its mask remains false.

Each window ends at the target time. Attention may use the earlier rows in that window; it must not receive scans after the target time. Do not use full-flight offline imputation to report this past-to-present outage result, because future healthy scans after recovery would change the information available to the model.

## Evaluate against persistence

Evaluate SAITS and `persistence_y` on exactly the same scored outage endpoints. Use `target_times_s` or `source_row_positions` to align window predictions with the prepared partition. Report distance-feature error in metres separately from valid-return-fraction error, since they have different units. State the number of scored timestamps and the outage intervals alongside the errors.

Retain healthy scans as input context, but do not average their easy reconstruction errors into the outage score. Report this as a one-flight feasibility result, with overlapping windows and a small held-out sample. A stronger claim requires additional flights and trajectory-level held-out evaluation.

## Primary references

- [Official SAITS repository](https://github.com/WenjieDu/SAITS): the model implementation and its original imputation setup. Check its expected inputs and training objective when writing the adapter above.
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): temporal ordering and why ordinary random folds are unsuitable for time-dependent observations. The helper here uses one fixed chronological split, not rolling cross-validation.
- [scikit-learn common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage): fitting preprocessing only on training data and keeping held-out data outside model selection.
