import os
import subprocess
import sys

def main():
    nodes = [
        "vTgJ", "zozf", "CA4O", 
        "lPxq", "wmrX", "h8bC", 
        "0D29", "47QD", "HAVn", 
        "BNuP", "r06G"
    ]
    
    raw_data_dir = "data/raw/measured/20260523_003445_ms-HvA1-HvA1"
    search_dir = "data/classification_output"
    
    # We will use the same python interpreter that runs this script
    python_exec = sys.executable
    
    for node in nodes:
        # Find the predictions.h5 for this node dynamically
        pred_h5 = None
        for root, dirs, files in os.walk(search_dir):
            if node in root and "predictions.h5" in files:
                pred_h5 = os.path.join(root, "predictions.h5")
                break
        
        if pred_h5 is None:
            print(f"[-] Could not find predictions.h5 for node '{node}'. Skipping...")
            continue
            
        print(f"\n=======================================================")
        print(f"[*] Running localise.py for node: {node}")
        print(f"[*] Predictions File: {pred_h5}")
        print(f"=======================================================")
        
        # Build the command using forward slashes for cross-platform compatibility
        pred_h5_norm = pred_h5.replace("\\", "/")
        
        cmd = [
            python_exec, 
            "src/localisation/localise.py",
            "--predictions_h5", pred_h5_norm,
            "--raw_data_dir", raw_data_dir,
            "--parent_node", node
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[+] Successfully completed node: {node}\n")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running node {node}. Error code: {e.returncode}\n")

if __name__ == "__main__":
    main()
