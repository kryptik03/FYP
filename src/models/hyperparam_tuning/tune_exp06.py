import os
import sys
import yaml
import copy
import argparse
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    print("[Error] Optuna is not installed. Run: pip install optuna")
    sys.exit(1)

# Add project root directory to path (3 levels up from this script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.models.data.dataset_exp06_dec import DECDataset_Exp06
from src.models.tasks.task_exp06_dec import SemiSupervisedDECTask

def build_loaders(cfg: dict):
    # Train loader
    train_dataset = DECDataset_Exp06(
        sources=cfg["data"]["sources"],
        shard_key="train_shards",
        max_pulse_len=cfg["data"]["max_pulse_len"],
        augment=True, # Always augment for Phase 1 SimCLR
        label_fraction=cfg["data"].get("label_fraction", 0.10)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True
    )

    # Val loader (no augmentation)
    val_dataset = DECDataset_Exp06(
        sources=cfg["data"]["sources"],
        shard_key="val_shards",
        max_pulse_len=cfg["data"]["max_pulse_len"],
        augment=False,
        label_fraction=cfg["data"].get("label_fraction", 0.10)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )
    
    return train_loader, val_loader

def objective(trial, base_config: dict):
    # 1. Suggest Hyperparameters
    cfg = copy.deepcopy(base_config)
    
    search_space = cfg.get("tune", {}).get("search_space", {})
    
    # Override Learning Rates
    # NOTE: Cast to float — YAML may parse scientific notation (e.g. 1e-5) as a string.
    if "learning_rate_phase1" in search_space:
        bounds = search_space["learning_rate_phase1"]
        lr1 = trial.suggest_float("learning_rate_phase1", float(bounds[0]), float(bounds[1]), log=True)
        cfg["training"]["learning_rate_phase1"] = lr1
        
    if "learning_rate_phase2" in search_space:
        bounds = search_space["learning_rate_phase2"]
        lr2 = trial.suggest_float("learning_rate_phase2", float(bounds[0]), float(bounds[1]), log=True)
        cfg["training"]["learning_rate_phase2"] = lr2
        
    # Override Pairwise Gamma
    if "pairwise_weight_gamma" in search_space:
        bounds = search_space["pairwise_weight_gamma"]
        gamma = trial.suggest_float("pairwise_weight_gamma", float(bounds[0]), float(bounds[1]), log=True)
        cfg["task"]["pairwise_weight_gamma"] = gamma
        
    # Override Base Channels
    if "base_channels" in search_space:
        options = [int(x) for x in search_space["base_channels"]]  # ensure int
        bc = trial.suggest_categorical("base_channels", options)
        cfg["model"]["base_channels"] = bc

    # 2. Setup Device & Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)
    
    if len(train_loader) == 0:
        raise ValueError("Train loader is empty. Check your paths.")

    # 3. Setup Model
    task = SemiSupervisedDECTask(cfg).to(device)

    # 4. Phase 1: SimCLR
    task.set_phase(1, lr=cfg["training"]["learning_rate_phase1"])
    for epoch in range(cfg["training"]["phase1_epochs"]):
        task.train()
        for batch in train_loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            task.training_step_phase1(batch)

    # 5. Extract Embeddings & Initialize K-Means
    task.eval()
    all_embs = []
    with torch.no_grad():
        for batch in train_loader:
            # For Phase 1 augmentation True, it returns view1, view2
            # We can just extract from view1
            view1 = batch[0].to(device)
            z = task.backbone(view1)
            all_embs.append(z.cpu().numpy())
            
    all_embs = np.concatenate(all_embs, axis=0)
    task.init_cluster_centroids(all_embs)
    
    # Update train_loader to no longer augment for Phase 2 
    # (Phase 2 DEC expects unaugmented single signal)
    # Actually, we can just use the first view of the augmented batch for speed
    
    # 6. Phase 2: Semi-Supervised DEC
    task.set_phase(2, lr=cfg["training"]["learning_rate_phase2"])
    
    best_val_loss = float('inf')
    
    for epoch in range(cfg["training"]["phase2_epochs"]):
        task.train()
        for batch in train_loader:
            # Batch has view1, view2, reported_class, ...
            # We treat view1 as the signal for DEC
            view1 = batch[0].to(device)
            reported_class = batch[2].to(device)
            dec_batch = (view1, reported_class)
            
            task.training_step_phase2(dec_batch)
            
        # Validation
        task.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                signal = batch[0].to(device)
                reported_class = batch[1].to(device)
                dec_batch = (signal, reported_class)
                
                res = task.validation_step(dec_batch)
                val_losses.append(res["total"])
                
        epoch_val_loss = np.mean(val_losses)
        best_val_loss = min(best_val_loss, epoch_val_loss)
        
        # Report intermediate value to Optuna for pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/models/configs/tune_exp06_dec.yaml")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Starting Optuna hyperparameter tuning for {config['experiment']['name']}")
    
    study = optuna.create_study(direction="minimize", study_name=config['experiment']['name'])
    n_trials = config.get("tune", {}).get("n_trials", 10)
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)
    
    best = study.best_trial
    print(f"\n[Tuning Complete] Best trial:")
    print(f"  Value (Validation Loss): {best.value}")
    print("  Params: ")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    # -----------------------------------------------------------------------
    # 1. Write best parameters back into the YAML config file
    # -----------------------------------------------------------------------
    # Re-read the YAML raw text so we preserve comments
    import json, string, random
    import ruamel.yaml

    ryaml = ruamel.yaml.YAML()
    ryaml.preserve_quotes = True
    with open(config_path, "r") as f:
        doc = ryaml.load(f)

    tune_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_node_id   = "".join(random.choices(string.ascii_letters + string.digits, k=4))

    # Overwrite training hyperparams that Optuna tuned
    if "learning_rate_phase1" in best.params:
        doc["training"]["learning_rate_phase1"] = round(best.params["learning_rate_phase1"], 10)
    if "learning_rate_phase2" in best.params:
        doc["training"]["learning_rate_phase2"] = round(best.params["learning_rate_phase2"], 10)

    # Overwrite model/task params that Optuna tuned
    if "base_channels" in best.params:
        doc["model"]["base_channels"] = int(best.params["base_channels"])
    if "pairwise_weight_gamma" in best.params:
        doc["task"]["pairwise_weight_gamma"] = round(best.params["pairwise_weight_gamma"], 6)

    # Stamp tuning metadata into the training block
    doc["training"]["tuned_at"]      = tune_timestamp
    doc["training"]["tuning_node_id"] = tune_node_id
    doc["training"]["tuning_val_loss"] = round(float(best.value), 8)

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)
    print(f"\n[Saved] Best params written back to: {config_path}")

    # -----------------------------------------------------------------------
    # 2. Register this tuning run in the lineage database
    # -----------------------------------------------------------------------
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.utils.lineage_tracker import register_process

    parent_id = config["experiment"].get("parent_node_id", "NONE")
    register_process(
        parent_id        = parent_id,
        stage            = "hyperparameter_tuning",
        method           = "optuna",
        folder_path      = config_path,
        appended_history = (
            f"Optuna tuning ({n_trials} trials). "
            f"Best val loss: {best.value:.6f}. "
            f"Params: {json.dumps(best.params)}"
        ),
        force_node_id    = tune_node_id,
        force_timestamp  = tune_timestamp,
    )
    print(f"[Lineage] Registered tuning node: {tune_node_id}")

    # -----------------------------------------------------------------------
    # 3. Write latest node ID to src/utils/latest_node.txt for Colab commits
    # -----------------------------------------------------------------------
    utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
    latest_node_path = os.path.join(utils_dir, "latest_node.txt")
    with open(latest_node_path, "w") as f:
        f.write(tune_node_id)
    print(f"[Lineage] Latest node ID written to: {latest_node_path}")

