import torch

print("ATTENTION MODEL:")
ckpt = torch.load("results/checkpoints/attention_cpu/attention/attention_20251123_143212_best.pth", map_location='cpu')
args = ckpt.get('args', {})
print(f"  Signal Length: {args.get('signal_length', 'NOT FOUND')}")
print(f"  Model Type: {args.get('model', 'NOT FOUND')}")
print(f"  Val Accuracy: {ckpt.get('val_acc', 'NOT FOUND')}")

print("\nCNN1D MODEL:")
ckpt = torch.load("results/checkpoints/cnn1d_cpu/cnn1d/cnn1d_20251123_083425_best.pth", map_location='cpu')
args = ckpt.get('args', {})
print(f"  Signal Length: {args.get('signal_length', 'NOT FOUND')}")
print(f"  Model Type: {args.get('model', 'NOT FOUND')}")
print(f"  Val Accuracy: {ckpt.get('val_acc', 'NOT FOUND')}")
