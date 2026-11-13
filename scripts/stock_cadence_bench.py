#!/usr/bin/env python3
"""Sustained run of one stock model at BirdNET-Go's analysis cadence: one
inference every STRIDE seconds for DURATION seconds (the way the detector
runs it), logging per-window latency, process CPU time, SoC temperature, CPU
clock and the firmware throttle flags. Shows whether a configuration keeps
up for minutes, not only in a 10-run burst.

usage: python -I stock_cadence_bench.py MODEL --threads N --stride 1.3333 \
           --duration 600 --npy W.npy --out OUT.json

Same session options as stock_threads_bench.py (ORT: intra-op = N, spinning
off; LiteRT: num_threads = N).
"""
import argparse
import json
import os
import statistics
import time

import numpy as np


def parse_args(argv=None):
    """The command line (sys.argv when argv is None)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stride", type=float, required=True)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def temp():
    """The SoC temperature in C (thermal zone 0)."""
    with open("/sys/class/thermal/thermal_zone0/temp", encoding="utf-8") as f:
        return int(f.read()) / 1000


def mhz():
    """The current clock of cpu0..cpu3 in MHz."""
    out = []
    for i in range(4):
        path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"
        with open(path, encoding="utf-8") as f:
            out.append(int(f.read()) // 1000)
    return out


def throttled():
    """The firmware's throttle flags (vcgencmd), empty when unavailable."""
    return os.popen("vcgencmd get_throttled 2>/dev/null").read().strip()


def litert_runner(model, threads):
    """run(x) on LiteRT with num_threads = threads."""
    # LiteRT is only needed for .tflite models: imported where used
    # pylint: disable-next=import-outside-toplevel
    from ai_edge_litert import interpreter
    it = interpreter.Interpreter(model_path=model, num_threads=threads)
    it.allocate_tensors()
    inp = it.get_input_details()[0]

    def run(x):
        it.set_tensor(inp["index"], x)
        it.invoke()
    return run


def ort_runner(model, threads):
    """run(x) on ONNX Runtime's CPU provider, BirdNET-Go's session options."""
    # ONNX Runtime is only needed for ONNX models: imported where used
    # pylint: disable-next=import-outside-toplevel
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name

    def run(x):
        s.run(None, {name: x})
    return run


def make_runner(model, threads):
    """run(x) for one window on the runtime the model file needs."""
    if model.endswith(".tflite"):
        return litert_runner(model, threads)
    return ort_runner(model, threads)


def minute_stats(windows, stride, t_s):
    """The last minute's median latency with the board's state."""
    w = windows[-int(60 / stride):]
    return {"t_s": round(t_s), "median_ms": round(statistics.median(w), 1),
            "temp_c": temp(), "mhz": mhz(), "throttled": throttled(),
            "load": os.getloadavg()[0]}


def cadence_loop(run, xs, stride, duration, log):
    """One window every stride seconds for duration seconds.

    Logs each window's latency and a per-minute summary into log.

    Returns:
        (wall seconds, process CPU seconds, late windows).
    """
    t_start = time.perf_counter()
    c_start = time.process_time()
    next_t = t_start
    i = 0
    late = 0
    last_min = t_start
    while time.perf_counter() - t_start < duration:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        t0 = time.perf_counter()
        run(xs[i % len(xs)])
        lat = (time.perf_counter() - t0) * 1e3
        log["windows"].append(round(lat, 1))
        i += 1
        next_t += stride
        if time.perf_counter() > next_t:
            late += 1
            # BirdNET-Go would run the next buffered window at once
            next_t = time.perf_counter()
        if time.perf_counter() - last_min >= 60:
            last_min = time.perf_counter()
            log["minutes"].append(minute_stats(log["windows"], stride,
                                               last_min - t_start))
            print(f"  {log['minutes'][-1]}", flush=True)
    wall = time.perf_counter() - t_start
    cpu = time.process_time() - c_start
    return wall, cpu, late


def summarize(log, late, wall, cpu):
    """The run's totals and distribution, added to log."""
    w = log["windows"]
    log.update({"n": len(w), "late_windows": late, "wall_s": round(wall, 1),
                "cpu_s": round(cpu, 1), "cores_busy": round(cpu / wall, 2),
                "median_ms": round(statistics.median(w), 1),
                "p95_ms": round(sorted(w)[int(0.95 * len(w))], 1),
                "max_ms": max(w), "temp_end": temp(),
                "throttled_end": throttled(),
                "duty": round(sum(w) / 1e3 / wall, 3)})


def main():
    a = parse_args()
    x = np.load(a.npy).astype(np.float32)
    xs = [np.ascontiguousarray(x[i][None, :]) for i in range(len(x))]
    run = make_runner(a.model, a.threads)
    run(xs[0])
    run(xs[1 % len(xs)])
    log = {"model": os.path.basename(a.model), "threads": a.threads,
           "stride_s": a.stride, "temp_start": temp(),
           "throttled_start": throttled(), "windows": [], "minutes": []}
    wall, cpu, late = cadence_loop(run, xs, a.stride, a.duration, log)
    summarize(log, late, wall, cpu)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"{log['model']} t={a.threads} stride {a.stride:.3f} s: "
          f"{len(log['windows'])} windows, median {log['median_ms']} ms, "
          f"p95 {log['p95_ms']}, max {log['max_ms']}, late {late}, "
          f"duty {log['duty']}, CPU {log['cores_busy']} cores, "
          f"temp {log['temp_start']} -> {log['temp_end']} C, "
          f"{log['throttled_end']}", flush=True)


if __name__ == "__main__":
    main()
