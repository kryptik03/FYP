import subprocess
import re
import sys

commands = [
    (
        "exp09", 
        r".venv\Scripts\python src/models/predictions/predict_exp09.py --checkpoint_id k7vt --grouping_mode channel_capped --reduce_method umap --reduce_dims 2 --time_threshold 1e-05 --dist_threshold 0.5 --source data/features/bispectra_v2/20260602_180648-ms-HvA1-RyNB:measured:all_shards"
    ),
    (
        "exp09", 
        r".venv\Scripts\python src/models/predictions/predict_exp09.py --checkpoint_id k7vt --grouping_mode channel_capped --reduce_method none --reduce_dims 2 --time_threshold 1e-05 --dist_threshold 0.5 --source data/features/bispectra_v2/20260602_180648-ms-HvA1-RyNB:measured:all_shards"
    )
]

results = []

for idx, (exp, cmd) in enumerate(commands, 10):
    print(f"\n[{idx}/11] Running {exp}...")
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
print("EXP09 RUNS COMPLETED")
print("="*80)
for res in results:
    print(res)

with open("batch_run_exp09_summary.txt", "w") as f:
    for res in results:
        f.write(res + "\n")
