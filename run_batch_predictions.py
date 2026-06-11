import subprocess
import re
import sys

commands = [
    (
        "exp06", 
        r".venv\Scripts\python src/models/predictions/predict_exp06.py --checkpoint_id M3Fu --grouping_mode channel_capped --source data/features/stft_magnitude/20260525_132349-ms-HvA1-kmqo:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method tsne --reduce_dims 2 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp06", 
        r".venv\Scripts\python src/models/predictions/predict_exp06.py --checkpoint_id M3Fu --grouping_mode channel_capped --source data/features/stft_magnitude/20260525_132349-ms-HvA1-kmqo:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method tsne --reduce_dims 3 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp06", 
        r".venv\Scripts\python src/models/predictions/predict_exp06.py --checkpoint_id M3Fu --grouping_mode channel_capped --source data/features/stft_magnitude/20260525_132349-ms-HvA1-kmqo:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method none --reduce_dims 2 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp07", 
        r".venv\Scripts\python src/models/predictions/predict_exp07.py --checkpoint_id ltlo --grouping_mode channel_capped --source data/features/stft_magnitude/20260525_132349-ms-HvA1-kmqo:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method tsne --reduce_dims 3 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp07", 
        r".venv\Scripts\python src/models/predictions/predict_exp07.py --checkpoint_id ltlo --grouping_mode channel_capped --source data/features/stft_magnitude/20260525_132349-ms-HvA1-kmqo:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method none --reduce_dims 2 --epsilon 0.01 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp07", 
        r".venv\Scripts\python src/models/predictions/predict_exp07.py --checkpoint_id ltlo --grouping_mode channel_capped --source data/features/stft_magnitude/20260525_132349-ms-HvA1-kmqo:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method tsne --reduce_dims 2 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp08", 
        r".venv\Scripts\python src/models/predictions/predict_exp08.py --checkpoint_id VNce --grouping_mode channel_capped --source data/features/bispectra/20260530_221215-ms-HvA1-qeub:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method tsne --reduce_dims 2 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp08", 
        r".venv\Scripts\python src/models/predictions/predict_exp08.py --checkpoint_id VNce --grouping_mode channel_capped --source data/features/bispectra/20260530_221215-ms-HvA1-qeub:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method tsne --reduce_dims 3 --epsilon 5.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp08", 
        r".venv\Scripts\python src/models/predictions/predict_exp08.py --checkpoint_id VNce --grouping_mode channel_capped --source data/features/bispectra/20260530_221215-ms-HvA1-qeub:measured:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --reduce_method none --reduce_dims 2 --epsilon 0.0 --min_cluster_size 5 --time_threshold 1e-07 --dist_threshold 0.5"
    ),
    (
        "exp09", 
        r".venv\Scripts\python src/models/predictions/predict_exp09.py --checkpoint_id k7vt --grouping_mode channel_capped --reduce_method umap --reduce_dims 2 --time_threshold 1e-05 --dist_threshold 0.5"
    ),
    (
        "exp09", 
        r".venv\Scripts\python src/models/predictions/predict_exp09.py --checkpoint_id k7vt --grouping_mode channel_capped --reduce_method none --reduce_dims 2 --time_threshold 1e-05 --dist_threshold 0.5"
    )
]

results = []

for idx, (exp, cmd) in enumerate(commands, 1):
    print(f"\n[{idx}/{len(commands)}] Running {exp}...")
    print(f"Command: {cmd}")
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_dir = None
    
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Look for the output directory in the logs
        if exp == "exp09":
            # [Output] D:\Zee_Documents\...
            match = re.search(r"\[Output\]\s+(.*)", line)
            if match:
                out_dir = match.group(1).strip()
        else:
            # [Done] Results saved -> D:\Zee_Documents\... or [Export] Saved pulse mappings -> ...
            match = re.search(r"Saved pulse mappings -> (.*?)\\predictions\.h5", line)
            if match:
                out_dir = match.group(1).strip()

    process.wait()
    
    if out_dir:
        results.append(f"Command {idx} ({exp}):\n  Run: {cmd}\n  Output Folder: {out_dir}\n")
    else:
        results.append(f"Command {idx} ({exp}):\n  Run: {cmd}\n  Output Folder: [Could not parse from logs]\n")

print("\n\n" + "="*80)
print("ALL RUNS COMPLETED")
print("="*80)
for res in results:
    print(res)

with open("batch_run_summary.txt", "w") as f:
    for res in results:
        f.write(res + "\n")
