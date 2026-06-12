"""
Deployment appendix (C5) — Convergence Plan P4.7.

Takes the benchmark's best single model (resnet18, by mean test accuracy),
and produces the honest deployment story:
  1. ONNX export (dynamic batch + length axes) + output-parity validation
  2. Dynamic INT8 quantization (onnxruntime)
  3. CPU latency table: torch fp32 vs onnx fp32 vs onnx int8,
     batch 1/8/32, p50/p95 over repeated runs
  4. FastAPI smoke: serve the ONNX artifact via the real API and verify
     /predict returns the correct class for a known test record

Artifacts: results/deployment/{latency.json,appendix.md} +
           checkpoints/deploy/resnet18_best.{onnx,int8.onnx}

Usage:
    python scripts/deployment_appendix.py
"""
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import NUM_CLASSES, FAULT_TYPES  # noqa: E402

WINDOW = 20480
CKPT = PROJECT_ROOT / 'results/benchmark/deep/resnet18/seed2/best_model.pth'
DEPLOY_DIR = PROJECT_ROOT / 'checkpoints/deploy'
OUT_DIR = PROJECT_ROOT / 'results/deployment'
REPEATS, WARMUP = 30, 5


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def bench(fn, x, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {'p50_ms': round(statistics.median(times), 2),
            'p95_ms': round(times[int(0.95 * len(times)) - 1], 2)}


def main() -> None:
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load best model -------------------------------------------------
    model = create_model('resnet18', num_classes=NUM_CLASSES)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded resnet18 seed2 (val {ckpt['best_val_acc']:.2f}%)")

    # ---- 1. ONNX export + parity -----------------------------------------
    onnx_path = DEPLOY_DIR / 'resnet18_best.onnx'
    dummy = torch.randn(1, 1, WINDOW)
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=14,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch', 2: 'length'},
                      'logits': {0: 'batch'}},
        dynamo=False,  # py3.14: dynamo exporter's onnxscript dep is broken
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    x = torch.randn(4, 1, WINDOW)
    with torch.no_grad():
        torch_out = model(x).numpy()
    onnx_out = sess.run(None, {'input': x.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff < 1e-3, f"ONNX parity failed: max diff {max_diff}"
    print(f"ONNX export OK (max |torch-onnx| = {max_diff:.2e}), "
          f"size {onnx_path.stat().st_size/1e6:.1f} MB")

    # ---- 2. dynamic INT8 quantization ------------------------------------
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path = DEPLOY_DIR / 'resnet18_best.int8.onnx'
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    sess8 = ort.InferenceSession(str(int8_path), providers=['CPUExecutionProvider'])
    int8_out = sess8.run(None, {'input': x.numpy()})[0]
    agree = float((int8_out.argmax(1) == torch_out.argmax(1)).mean())
    print(f"INT8 quantization OK, size {int8_path.stat().st_size/1e6:.1f} MB, "
          f"argmax agreement on random batch: {agree:.2f}")

    # ---- 3. latency table --------------------------------------------------
    latency = {}
    for batch in (1, 8, 32):
        xb = torch.randn(batch, 1, WINDOW)
        xb_np = xb.numpy()
        with torch.no_grad():
            latency[f'torch_fp32_b{batch}'] = bench(
                lambda t: model(t), xb)
        latency[f'onnx_fp32_b{batch}'] = bench(
            lambda t: sess.run(None, {'input': t}), xb_np)
        latency[f'onnx_int8_b{batch}'] = bench(
            lambda t: sess8.run(None, {'input': t}), xb_np)
        print(f"batch {batch}: torch {latency[f'torch_fp32_b{batch}']['p50_ms']}ms | "
              f"onnx {latency[f'onnx_fp32_b{batch}']['p50_ms']}ms | "
              f"int8 {latency[f'onnx_int8_b{batch}']['p50_ms']}ms (p50)")

    # ---- 4. FastAPI smoke with a real test record --------------------------
    import os
    os.environ['MODEL_PATH'] = str(onnx_path)
    os.environ['MODEL_TYPE'] = 'onnx'
    import h5py
    with h5py.File(PROJECT_ROOT / 'data/generated/dataset_v2.h5') as f:
        sig = f['test']['signals'][0].astype(float)
        true_label = int(f['test']['labels'][0])

    api_result = {'attempted': True}
    try:
        from fastapi.testclient import TestClient
        from packages.deployment.api import main as api_main
        import importlib
        importlib.reload(api_main)  # pick up MODEL_PATH env
        with TestClient(api_main.app) as client:
            r = client.post('/predict', json={'signal': sig.tolist(),
                                              'return_probabilities': False})
            api_result['status_code'] = r.status_code
            if r.status_code == 200:
                body = r.json()
                api_result['predicted_class'] = body['predicted_class']
                api_result['true_class'] = true_label
                api_result['correct'] = body['predicted_class'] == true_label
                print(f"API smoke: /predict -> {body['predicted_class']} "
                      f"({FAULT_TYPES[body['predicted_class']]}), "
                      f"true {true_label} ({FAULT_TYPES[true_label]}), "
                      f"correct={api_result['correct']}")
            else:
                api_result['detail'] = r.text[:300]
                print(f"API smoke: status {r.status_code} — {r.text[:200]}")
    except Exception as e:  # noqa: BLE001 — appendix must report, not crash
        api_result['error'] = f'{type(e).__name__}: {e}'
        print(f"API smoke failed: {api_result['error']}")

    # ---- artifacts ----------------------------------------------------------
    artifact = {
        'model': 'resnet18 (benchmark best single model, seed 2)',
        'checkpoint': str(CKPT.relative_to(PROJECT_ROOT)),
        'window_length': WINDOW,
        'onnx': {'path': str(onnx_path.relative_to(PROJECT_ROOT)),
                 'size_mb': round(onnx_path.stat().st_size / 1e6, 1),
                 'parity_max_abs_diff': max_diff},
        'int8': {'path': str(int8_path.relative_to(PROJECT_ROOT)),
                 'size_mb': round(int8_path.stat().st_size / 1e6, 1),
                 'argmax_agreement_random_batch': agree},
        'latency_ms': latency,
        'api_smoke': api_result,
        'provenance': {'host': platform.node(), 'git_sha': git_sha(),
                       'repeats': REPEATS,
                       'finished_at': datetime.now(timezone.utc).isoformat()},
    }
    (OUT_DIR / 'latency.json').write_text(json.dumps(artifact, indent=2))

    lines = ['# Deployment Appendix (C5)', '',
             f"Model: **resnet18** (best single benchmark model, 96.14% mean test acc)", '',
             '| Backend | Size | b=1 p50 | b=8 p50 | b=32 p50 |',
             '|---|---|---|---|---|']
    for name, key in [('Torch FP32', 'torch_fp32'), ('ONNX FP32', 'onnx_fp32'),
                      ('ONNX INT8', 'onnx_int8')]:
        size = ('—' if 'torch' in key else
                f"{artifact['onnx' if 'fp32' in key else 'int8']['size_mb']} MB")
        lines.append(f"| {name} | {size} | "
                     f"{latency[key + '_b1']['p50_ms']} ms | "
                     f"{latency[key + '_b8']['p50_ms']} ms | "
                     f"{latency[key + '_b32']['p50_ms']} ms |")
    lines += ['', f"ONNX parity: max |Δ| = {max_diff:.2e}; "
                  f"INT8 argmax agreement: {agree:.2f}",
              f"API smoke: {json.dumps(api_result)}"]
    (OUT_DIR / 'appendix.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nArtifacts -> {OUT_DIR}")


if __name__ == '__main__':
    main()
