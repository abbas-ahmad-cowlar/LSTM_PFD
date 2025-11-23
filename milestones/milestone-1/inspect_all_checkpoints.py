import torch
from pathlib import Path
import pprint

checkpoint_dir = Path("results/checkpoints")

print("=" * 80)
print("MODEL CHECKPOINT INVENTORY")
print("=" * 80)

# Find all best checkpoints
checkpoints = list(checkpoint_dir.rglob("*best.pth"))

if not checkpoints:
    print("No checkpoints found!")
else:
    for ckpt_path in sorted(checkpoints):
        print(f"\n{'='*80}")
        print(f"Checkpoint: {ckpt_path.relative_to(checkpoint_dir)}")
        print(f"{'='*80}")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Extract key info
            if 'args' in checkpoint:
                args = checkpoint['args']
                print(f"\nModel Type: {args.get('model', 'unknown')}")
                print(f"Signal Length: {args.get('signal_length', 'unknown')}")
                print(f"Epochs Trained: {checkpoint.get('epoch', 'unknown') + 1 if 'epoch' in checkpoint else 'unknown'}")
                print(f"Best Val Accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}" if 'val_acc' in checkpoint else "Val Acc: unknown")
                print(f"Data Directory: {args.get('data_dir', 'unknown')}")
                print(f"Batch Size: {args.get('batch_size', 'unknown')}")
                
                print(f"\nFull Training Args:")
                pprint.pprint(args, width=80, compact=True)
            else:
                print("No args found in checkpoint")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

print(f"\n{'='*80}")
print(f"Total checkpoints found: {len(checkpoints)}")
print(f"{'='*80}")
