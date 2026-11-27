"""Measure a real CUDA training step on disposable weights before choosing batch size."""
from __future__ import annotations

import copy
import gc
import json
import math
from pathlib import Path
import time

from checkpoint_resume import capture_rng, restore_rng


def search_batch_size(probe, *, start, maximum, quantum=8):
    """Exponential search followed by refinement; probe reports fit and memory."""
    if start < 1 or maximum < 1 or quantum < 1:
        raise ValueError("Batch search bounds must be positive")
    trials, low, high = [], 0, None
    candidate = min(start, maximum)
    while True:
        result = dict(probe(candidate), batch_size=candidate)
        trials.append(result)
        if not result["fits"]:
            high = candidate
            break
        low = candidate
        if low == maximum:
            return low, trials
        candidate = min(maximum, candidate*2)
    if low == 0:
        candidate = max(1, high//2)
        while candidate >= 1:
            result = dict(probe(candidate), batch_size=candidate)
            trials.append(result)
            if result["fits"]:
                low = candidate
                break
            high = candidate
            if candidate == 1:
                raise RuntimeError("Even batch size 1 does not fit the GPU memory budget")
            candidate = max(1, candidate//2)
    while high-low > quantum:
        middle = ((low+high)//(2*quantum))*quantum
        middle = max(low+1, min(high-1, middle))
        result = dict(probe(middle), batch_size=middle)
        trials.append(result)
        if result["fits"]:
            low = middle
        else:
            high = middle
    return low, trials


def tune_batch_size(namespace, dataset, config, saved, report_path, *, fraction=.9,
                    maximum=1024, steps=2, control=None):
    """Preserve all production weights/RNGs. Include gradients and Adam state in probes.

    Targets and windows come from the real dataset; half the inputs are made
    full-blackout inputs to exercise that branch. Scratch optimizer steps never
    update a checkpoint or the eventual training model. No LR scaling is done.
    """
    import torch
    from torch.utils.data import default_collate

    if not .5 <= fraction <= .95 or maximum < 1 or steps < 1:
        raise ValueError("Use a memory fraction from 0.5 to 0.95 and positive probe sizes")
    device = namespace["runtime_device"](config)
    if device.type != "cuda":
        raise ValueError("Automatic batch sizing requires a CUDA GPU")
    state = capture_rng()
    t, r = config["training"], config["runtime"]
    old_benchmark = torch.backends.cudnn.benchmark
    old_precision = torch.get_float32_matmul_precision()
    torch.backends.cudnn.benchmark = r["cudnn_benchmark"]
    torch.set_float32_matmul_precision(r["matmul_precision"])
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    external_bytes = total-free


    budget = int(total*fraction)-external_bytes
    if budget <= 0:
        raise RuntimeError("Insufficient free GPU memory for the requested budget")
    ordinal = max(range(len(dataset.records)), key=lambda i: dataset.records[i]["count"])
    start = int(dataset.offsets[ordinal])
    count = dataset.records[ordinal]["count"]
    pool = default_collate([dataset[start+min(count-1, max(config["data"]["history_steps"], count//2)+i)] for i in range(4)])
    report = dict(gpu=torch.cuda.get_device_name(device), total_gpu_gib=total/2**30,
                  free_gpu_gib=free/2**30, target_device_fraction=fraction,
                  torch_budget_gib=budget/2**30, maximum_batch_size=maximum,
                  previous_batch_size=t["batch_size"], amp_dtype=r["amp_dtype"], trials=[])

    def persist():
        path = Path(report_path)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        temporary.replace(path)

    def probe(batch_size):
        if control is not None and control.should_stop(stage="batch_calibration"):
            raise RuntimeError("Allocation time budget reached during batch calibration. The original training checkpoint is unchanged; retry in the next allocation.")
        model = optimizer = scaler = batch = output = numerator = denominator = loss = None
        try:
            gc.collect(); torch.cuda.empty_cache()
            model = namespace["SparseRangeReconstructor"](config["model"], config["data"]["sample_period_s"]).to(device)
            if saved is not None:
                model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
                betas=tuple(t["betas"]), eps=t["eps"], fused=r["fused_adamw"])
            if saved is not None:
                optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state_dict"]))
            scaler = torch.amp.GradScaler("cuda", enabled=r["amp_dtype"] == "fp16")
            if saved is not None:
                scaler.load_state_dict(saved["scaler_state_dict"])
            indices = torch.arange(batch_size) % len(pool["ranges"])
            batch = {k: v.index_select(0, indices).to(device) for k, v in pool.items()}
            batch["available"][::2] = False
            batch["valid"][::2] = False
            batch["ranges"][::2] = 0
            batch["history"][::2] = 0
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            def update():
                nonlocal output, numerator, denominator, loss
                optimizer.zero_grad(set_to_none=True)
                with namespace["autocast_context"](config, device):
                    output = model(**namespace["model_inputs"](batch))
                numerator, denominator = namespace["range_weighted_terms"](output, batch["target"], batch["target_valid"], **namespace["loss_options"](config))
                loss = numerator/denominator
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite calibration loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
            update()  # allocate optimizer state and warm convolution algorithms
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(steps):
                update()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-started
            peak = torch.cuda.max_memory_reserved(device)
            free_now, _ = torch.cuda.mem_get_info(device)
            outside_pool = max(external_bytes, total-free_now-torch.cuda.memory_reserved(device))
            allowed = min(budget, int(total*fraction)-outside_pool)
            result = dict(fits=peak <= allowed, peak_reserved_gib=peak/2**30,
                          estimated_peak_device_fraction=(peak+outside_pool)/total,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                          samples_per_second=batch_size*steps/elapsed,
                          seconds_per_step=elapsed/steps)
        except torch.cuda.OutOfMemoryError:
            result = dict(fits=False, reason="CUDA out of memory")
        finally:
            model = optimizer = scaler = batch = output = numerator = denominator = loss = None
            gc.collect(); torch.cuda.empty_cache()
        report["trials"].append(dict(batch_size=batch_size, **result))
        persist()
        print("GPU batch probe:", batch_size, result, flush=True)
        return result

    try:
        selected, _ = search_batch_size(probe, start=max(16, t["batch_size"]), maximum=maximum)
        # Recheck the selected shape after all the larger probes/OOM cleanup.
        confirmation = probe(selected)
        while not confirmation["fits"] and selected > 1:
            selected = max(1, int(selected*.8))
            confirmation = probe(selected)
        if not confirmation["fits"]:
            raise RuntimeError("No stable batch size fits the memory budget")
        report.update(selected_batch_size=selected, status="completed",
                      note="Disposable training steps only. Real weights/optimizer are restored from the original checkpoint. GPU throughput excludes host data loading.")
        persist()
        print(f"Selected batch size {selected} for {report['gpu']} ({total/2**30:.1f} GiB); report: {report_path}", flush=True)
        return selected, report
    except BaseException as exc:
        report.update(status="failed", error=str(exc)); persist()
        raise
    finally:
        dataset.close()
        torch.backends.cudnn.benchmark = old_benchmark
        torch.set_float32_matmul_precision(old_precision)
        restore_rng({"rng_state": state})
        gc.collect(); torch.cuda.empty_cache()
