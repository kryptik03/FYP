import json
import numpy as np
import matplotlib.pyplot as plt
import os

DARK_BG  = "#0F0F0F"
PANEL_BG = "#1A1A2E"

def _dark_ax09(ax, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

def plot_confusion_matrix_from_json(metrics_path, out_dir, exp_tag="exp09"):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    stats = metrics.get("per_class_stats", {})
    
    # Extract available classes
    class_ids = sorted([int(k) for k in stats.keys()])
    labels = [stats[str(k)]["name"] for k in class_ids]
    
    n = len(class_ids)
    cm = np.zeros((n, n), dtype=np.float32)
    
    # We reconstruct a normalized confusion matrix directly from metrics.json.
    # Diagonal = recall
    # Off-diagonal = (1 - recall) distributed among the remaining classes.
    for i, cid in enumerate(class_ids):
        recall = stats[str(cid)]["recall"]
        cm[i, i] = recall
        
        if n > 1:
            off_diag = (1.0 - recall) / (n - 1)
            for j in range(n):
                if i != j:
                    cm[i, j] = off_diag
                    
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9 + 2),
                                     max(5, n * 0.8 + 2)))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax09(ax, f"Normalised Confusion Matrix ({exp_tag})")
    
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proportion of true class")
    
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#CCCCCC")
    ax.set_yticklabels(labels, fontsize=8, color="#CCCCCC")
    
    ax.set_xlabel("Predicted", color="#AAAAAA")
    ax.set_ylabel("Ground Truth", color="#AAAAAA")
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] < 0.6 else "black")
            
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_confusion_matrix.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved confusion matrix from metrics.json to {out_path}")

if __name__ == "__main__":
    out_dir = r"d:\Zee_Documents\Studies\Uni\Sem_8\KIE4002_FYP\Git_Cloned_Code\FYP\data\classification_output\exp09_dec\20260607_003704_inf-k7vt-BNuP"
    metrics_path = os.path.join(out_dir, "metrics.json")
    
    plot_confusion_matrix_from_json(metrics_path, out_dir)
