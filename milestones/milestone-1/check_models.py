import torch
from pathlib import Path

checkpoint_dir = Path("results/checkpoints")
checkpoints = list(checkpoint_dir.rglob("*best.pth"))

print("\nMODEL CHECKPOINT SUMMARY")
print("=" * 80)

for ckpt_path in sorted(checkpoints):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    args = checkpoint.get('args', {})
    
    print(f"\nFile: {ckpt_path.name}")
    print(f"  Model: {args.get('model', 'unknown')}")
    print(f"  Signal Length: {args.get('signal_length', 'unknown')}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 0):.2%}")
    print(f"  Epoch: {checkpoint.get('epoch', -1) + 1}")

print(f"\n{'='*80}")
print(f"Total: {len(checkpoints)} checkpoints")
