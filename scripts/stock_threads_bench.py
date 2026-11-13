#!/usr/bin/env python3
"""Stock-runtime latency, CPU time and peak RSS of one model at one thread
count, in a fresh process.

usage: python -I stock_threads_bench.py MODEL --threads N --npy WINDOWS.npy \
           --out OUT.json [--runs 10] [--warm 2]

ONNX models run on ONNX Runtime's CPU execution provider with the session
options BirdNET-Go uses for BirdNET v3.0 (intra-op threads = N, spinning off,
sequential execution, default graph optimization); .tflite models run on
LiteRT with num_threads = N (XNNPACK delegate, default flags); --ov runs an
ONNX model on OpenVINO CPU (f32). Each timed run uses the next window of
WINDOWS.npy (cycling).
Reported per run: wall time and process CPU time (user + system over all
threads). Peak RSS = VmHWM of this process (the model session plus the
Python interpreter and numpy, ~30 MB). Run one model per process so the peak
is the model's own. The full outputs of the first two windows are stored
(for cross-thread agreement checks).
"""
import argparse
import importlib.metadata
import json
import os
import resource
import statistics
import time

import numpy as np

EMBEDDING_WIDTH = 1280  # the output that is not this wide is the predictions


def parse_args(argv=None):
    """The command line (sys.argv when argv is None)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--ov", action="store_true",
                    help="ONNX on OpenVINO CPU, f32 precision hint, "
                         "INFERENCE_NUM_THREADS = N")
    return ap.parse_args(argv)


def status(key):
    """A /proc/self/status value in MB (None when absent)."""
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0
    return None


def read_int(path):
    """An integer from a sysfs file."""
    with open(path, encoding="utf-8") as f:
        return int(f.read())


def sysinfo():
    """Load, temperature, clocks, throttle flags, affinity, free memory."""
    d = {"load": list(os.getloadavg())}
    try:
        d["temp_c"] = read_int("/sys/class/thermal/thermal_zone0/temp") / 1000
        d["mhz"] = [read_int(f"/sys/devices/system/cpu/cpu{i}/cpufreq/"
                             "scaling_cur_freq") // 1000
                    for i in range(os.cpu_count())]
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # no sysfs thermal / cpufreq nodes on this host: left out
    try:
        flags = os.popen("vcgencmd get_throttled 2>/dev/null").read()
        d["throttled"] = flags.strip()
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # no shell / popen here: left out
    d["affinity"] = sorted(os.sched_getaffinity(0))
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                d["mem_available_mb"] = int(line.split()[1]) // 1024
    return d


def prediction_size(o):
    """Sort key: the element count of an output unless it is the embedding."""
    if o["shape"][-1] != EMBEDDING_WIDTH:
        return int(np.prod(o["shape"]))
    return -1


def litert_runner(model, threads):
    """(run, runtime name) on LiteRT; run(x) returns the predictions."""
    # LiteRT is only needed for .tflite models: imported where used
    # pylint: disable-next=import-outside-toplevel
    from ai_edge_litert import interpreter
    it = interpreter.Interpreter(model_path=model, num_threads=threads)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    oi = max(outs, key=prediction_size)
    runtime = "litert " + importlib.metadata.version("ai-edge-litert")

    def run(x):
        it.set_tensor(inp["index"], x)
        it.invoke()
        return it.get_tensor(oi["index"]).copy()
    return run, runtime


def openvino_runner(model, threads):
    """(run, runtime name) on OpenVINO CPU, f32; run(x) returns predictions."""
    # OpenVINO is only needed with --ov: imported where used
    # pylint: disable-next=import-outside-toplevel
    import openvino as ov
    core = ov.Core()
    cm = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32",
                                          "INFERENCE_NUM_THREADS": threads,
                                          "PERFORMANCE_HINT": "LATENCY"})
    req = cm.create_infer_request()
    oport = [o for o in cm.outputs
             if o.get_partial_shape()[-1].get_length() != EMBEDDING_WIDTH][0]
    runtime = "openvino " + ov.__version__.split("-")[0]

    def run(x):
        return req.infer({0: x})[oport].copy()
    return run, runtime


def ort_runner(model, threads):
    """(run, runtime name) on ONNX Runtime's CPU provider with BirdNET-Go's
    session options; run(x) returns the predictions."""
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
    # predictions = the output that is not the 1280-wide embedding
    out_name = [o.name for o in s.get_outputs()
                if o.shape[-1] != EMBEDDING_WIDTH][0]
    runtime = "ort " + ort.__version__

    def run(x):
        return s.run([out_name], {name: x})[0]
    return run, runtime


def make_runner(model, threads, use_ov):
    """(run, runtime name) on the runtime the model file (or --ov) needs."""
    if model.endswith(".tflite"):
        return litert_runner(model, threads)
    if use_ov:
        return openvino_runner(model, threads)
    return ort_runner(model, threads)


def timed_runs(run, xs, runs):
    """(wall ms, cpu ms) per run over the windows (cycling)."""
    wall = []
    cpu = []
    for i in range(runs):
        c0, t0 = time.process_time(), time.perf_counter()
        run(xs[i % len(xs)])
        wall.append((time.perf_counter() - t0) * 1e3)
        cpu.append((time.process_time() - c0) * 1e3)
    return wall, cpu


def add_stats(info, wall, cpu):
    """The timing distribution, memory and the board's state after the run."""
    info["wall_ms"] = {"median": round(statistics.median(wall), 1),
                       "min": round(min(wall), 1), "max": round(max(wall), 1),
                       "all": [round(v, 1) for v in wall]}
    info["cpu_ms"] = {"median": round(statistics.median(cpu), 1),
                      "all": [round(v, 1) for v in cpu]}
    info["vmhwm_mb"] = round(status("VmHWM"), 1)
    info["vmrss_mb"] = round(status("VmRSS"), 1)
    info["ru_maxrss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    info["sys_after"] = sysinfo()


def main():
    a = parse_args()
    x = np.load(a.npy).astype(np.float32)
    xs = [np.ascontiguousarray(x[i][None, :]) for i in range(len(x))]
    info = {"model": os.path.basename(a.model), "threads": a.threads,
            "sys_before": sysinfo()}
    t_load = time.perf_counter()
    run, runtime = make_runner(a.model, a.threads, a.ov)
    info["runtime"] = runtime
    info["load_s"] = round(time.perf_counter() - t_load, 2)
    outs0 = [run(xs[i % len(xs)]) for i in range(max(a.warm, 2))]
    wall, cpu = timed_runs(run, xs, a.runs)
    add_stats(info, wall, cpu)
    np.save(a.out.replace(".json", "_out.npy"),
            np.concatenate([np.asarray(o).reshape(1, -1) for o in outs0[:2]]))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(f"{info['model']:58s} {info['runtime']:13s} t={a.threads} "
          f"wall {info['wall_ms']['median']:8.1f} ms "
          f"(min {info['wall_ms']['min']:.1f})  "
          f"cpu {info['cpu_ms']['median']:8.1f} ms  "
          f"VmHWM {info['vmhwm_mb']:.0f} MB  "
          f"load {info['sys_before']['load'][0]:.2f}->"
          f"{info['sys_after']['load'][0]:.2f}  "
          f"T {info['sys_after'].get('temp_c')}  "
          f"MHz {info['sys_after'].get('mhz')} "
          f"{info['sys_after'].get('throttled')}", flush=True)


if __name__ == "__main__":
    main()
