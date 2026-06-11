import os
import h5py

raw_dir = 'data/raw'

# Dict to hold results: dataset_name -> { 'shards': 0, 'scenes': 0, 'signals': 0 }
stats = {}

for root, dirs, files in os.walk(raw_dir):
    h5_files = [f for f in files if f.endswith('.h5')]
    if h5_files:
        # The dataset name is the directory name, optionally prefix it with the parent directory
        parent = os.path.basename(os.path.dirname(root))
        dataset_name = f"{parent}/{os.path.basename(root)}"
        
        stats[dataset_name] = {'shards': len(h5_files), 'scenes': 0, 'signals': 0}
        
        for f in h5_files:
            path = os.path.join(root, f)
            try:
                with h5py.File(path, 'r') as hf:
                    if 'scenes' in hf:
                        shape = hf['scenes'].shape
                        num_scenes = shape[0]
                        num_channels = shape[1]
                        
                        stats[dataset_name]['scenes'] += num_scenes
                        stats[dataset_name]['signals'] += (num_scenes * num_channels)
            except Exception as e:
                pass

print(f"{'Dataset Category/Name':<50} | {'Shards':<8} | {'Scenes':<8} | {'Signals':<8}")
print("-" * 81)
for ds, data in sorted(stats.items()):
    print(f"{ds:<50} | {data['shards']:<8} | {data['scenes']:<8} | {data['signals']:<8}")
