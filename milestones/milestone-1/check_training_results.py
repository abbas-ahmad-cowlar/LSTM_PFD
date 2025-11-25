import torch
from pathlib import Path

checkpoints = list(Path('results/checkpoints_full').rglob('*_best.pth'))

print('\n' + '='*60)
print('TRAINING RESULTS SUMMARY')
print('='*60 + '\n')

for cp in sorted(checkpoints):
    try:
        checkpoint = torch.load(cp, map_location='cpu')
        model_name = cp.parent.name.upper()
        val_acc = checkpoint.get('best_val_acc', 0)
        epoch = checkpoint.get('epoch', 'N/A')
        
        print(f'{model_name}:')
        print(f'  Checkpoint: {cp.name}')
        print(f'  Best Val Accuracy: {val_acc:.2f}%')
        print(f'  Epoch: {epoch}')
        print(f'  File: {cp}')
        print()
    except Exception as e:
        print(f'Error loading {cp}: {e}\n')

print('='*60)
