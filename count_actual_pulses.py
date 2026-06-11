import os
import h5py

raw_dir = 'data/raw'
stats = {}

for root, dirs, files in os.walk(raw_dir):
    h5_files = [f for f in files if f.endswith('.h5')]
    if h5_files:
        parent = os.path.basename(os.path.dirname(root))
        dataset_name = f"{parent}/{os.path.basename(root)}"
        
        stats[dataset_name] = {'shards': len(h5_files), 'scenes': 0, 'raw_channels': 0, 'actual_pulses': 0}
        
        for f in h5_files:
            path = os.path.join(root, f)
            try:
                with h5py.File(path, 'r') as hf:
                    if 'scenes' in hf:
                        num_scenes = hf['scenes'].shape[0]
                        num_channels = hf['scenes'].shape[1]
                        stats[dataset_name]['scenes'] += num_scenes
                        stats[dataset_name]['raw_channels'] += (num_scenes * num_channels)
                    
                    if 'labels' in hf:
                        # labels shape can be (features, num_pulses) or (num_pulses, features) depending on how it was saved
                        shape = hf['labels'].shape
                        # find which dimension is the number of pulses
                        # usually it's the larger dimension, or if it's (7, N) then N
                        if len(shape) == 2:
                            num_pulses = max(shape)
                            stats[dataset_name]['actual_pulses'] += num_pulses
            except Exception as e:
                pass

print(f"{'Dataset Category/Name':<50} | {'Scenes':<8} | {'Actual Pulses Injected / Detected':<35}")
print("-" * 100)
for ds, data in sorted(stats.items()):
    print(f"{ds:<50} | {data['scenes']:<8} | {data['actual_pulses']:<35}")
