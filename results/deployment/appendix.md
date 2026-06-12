# Deployment Appendix (C5)

Model: **resnet18** (best single benchmark model, 96.14% mean test acc)

| Backend | Size | b=1 p50 | b=8 p50 | b=32 p50 |
|---|---|---|---|---|
| Torch FP32 | — | 19.05 ms | 144.64 ms | 558.42 ms |
| ONNX FP32 | 15.4 MB | 12.92 ms | 101.06 ms | 417.96 ms |
| ONNX INT8 | 3.9 MB | 215.18 ms | 1564.94 ms | 6136.14 ms |

ONNX parity: max |Δ| = 1.53e-04; INT8 argmax agreement: 1.00
API smoke: {"attempted": true, "status_code": 200, "predicted_class": 0, "true_class": 0, "correct": true}