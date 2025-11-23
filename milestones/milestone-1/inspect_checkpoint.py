import torch
import sys
import pprint

checkpoint_path = "results/checkpoints/cnn1d_cpu/cnn1d/cnn1d_20251123_083425_best.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'args' in checkpoint:
        print("Training Arguments:")
        pprint.pprint(checkpoint['args'])
    else:
        print("No args found in checkpoint.")

except Exception as e:
    print(f"Error: {e}")
