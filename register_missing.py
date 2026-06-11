import os
import json
import sys

sys.path.append(os.path.join(os.getcwd(), 'src', 'utils'))
from lineage_tracker import register_process, get_node_history

folders = [
    r"D:\Zee_Documents\Studies\Uni\Sem_8\KIE4002_FYP\Git_Cloned_Code\FYP\data\classification_output\exp09_dec\20260607_003704_inf-k7vt-BNuP",
    r"D:\Zee_Documents\Studies\Uni\Sem_8\KIE4002_FYP\Git_Cloned_Code\FYP\data\classification_output\exp09_dec\20260607_004542_inf-k7vt-r06G"
]

for out_dir in folders:
    inf_id = out_dir[-4:]
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    node_id = metrics["checkpoint_id"]
    grouping_mode = metrics.get("grouping_mode", "channel_capped")

    history_str = (
        f"Exp09 inference (checkpoint={node_id}). "
        f"cls_acc={metrics.get('classification_accuracy', 0):.3f}, "
        f"cls_f1_macro={metrics.get('cls_f1_macro', 0):.3f}, "
        f"grouping_f1={metrics.get('grouping_f1', 0):.3f}, "
        f"auc_roc={metrics.get('auc_roc_macro')}, "
        f"silhouette={metrics.get('silhouette_score')}, "
        f"grouping_mode={grouping_mode}, "
        f"n_clusters={metrics.get('n_clusters_found')}."
    )

    register_process(
        parent_id        = node_id,
        stage            = "inference",
        method           = "dann_vit_umap_hdbscan_exp09",
        folder_path      = out_dir,
        appended_history = history_str,
        force_node_id    = inf_id,
    )
    
    with open(os.path.join(out_dir, "analysis_history.txt"), "w") as f:
        f.write(get_node_history(inf_id))

    print(f"Registered {inf_id} successfully.")
