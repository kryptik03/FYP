# src/utils/lineage_tracker.py

import sqlite3
import os
import shutil
from datetime import datetime
import string
import random

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/lineage.db'))

def init_db():
    """Initializes the SQLite database and the Edge List table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the Edge List Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            parent_id TEXT,
            root_id TEXT,
            origin TEXT,
            stage TEXT,
            method TEXT,
            folder_path TEXT,
            nickname TEXT,
            timestamp TEXT,
            history_log TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"[Tracker] Database initialized at {DB_PATH}")

def generate_short_id(length=4):
    """Generates a random 4-character alphanumeric ID."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def register_root_dataset(origin, method, folder_path, nickname, history_log, force_root_id=None, force_timestamp=None):
    """Registers a brand new synthetic or measured dataset (Root Node)."""
    # Accept MATLAB's ID and timestamp if provided, otherwise generate new ones
    root_id = force_root_id if force_root_id else generate_short_id()
    timestamp = force_timestamp if force_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO nodes (node_id, parent_id, root_id, origin, stage, method, folder_path, nickname, timestamp, history_log)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (root_id, "NONE", root_id, origin, "generation", method, folder_path, nickname, timestamp, history_log))
    
    conn.commit()
    conn.close()
    print(f"[Tracker] Root Dataset Registered: {root_id} ({nickname})")
    return root_id

def register_process(parent_id, stage, method, folder_path, appended_history):
    """Registers a downstream process (Child Node) linked to a parent."""
    node_id = generate_short_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch parent details to inherit origin, root_id, nickname, and previous history
    cursor.execute("SELECT root_id, origin, nickname, history_log FROM nodes WHERE node_id=?", (parent_id,))
    parent_data = cursor.fetchone()
    
    if not parent_data:
        raise ValueError(f"Parent Node {parent_id} does not exist in the database.")
        
    root_id, origin, nickname, parent_history = parent_data
    new_history = f"{parent_history} -> {appended_history}"
    
    cursor.execute('''
        INSERT INTO nodes (node_id, parent_id, root_id, origin, stage, method, folder_path, nickname, timestamp, history_log)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (node_id, parent_id, root_id, origin, stage, method, folder_path, nickname, timestamp, new_history))
    
    conn.commit()
    conn.close()
    print(f"[Tracker] Process Registered: Node {node_id} (Child of {parent_id})")
    return node_id

def trace_lineage(node_id):
    """Prints the full history of a node from its birth."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT history_log, folder_path FROM nodes WHERE node_id=?", (node_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        print(f"\n--- Lineage for Node {node_id} ---")
        print(f"Path: {result[1]}")
        print(f"History: {result[0]}\n")
    else:
        print(f"[Error] Node {node_id} not found.")

def prune_node(node_id):
    """Safely deletes a node's folder and removes it from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Check if it has children (Prevent breaking the DAG)
    cursor.execute("SELECT node_id FROM nodes WHERE parent_id=?", (node_id,))
    children = cursor.fetchall()
    if children:
        print(f"[Warning] Cannot prune {node_id}. It has downstream dependents: {[c[0] for c in children]}")
        print("Prune the children first.")
        conn.close()
        return

    # 2. Get folder path
    cursor.execute("SELECT folder_path FROM nodes WHERE node_id=?", (node_id,))
    result = cursor.fetchone()
    
    if not result:
        print(f"[Error] Node {node_id} not found.")
        conn.close()
        return
        
    folder_path = result[0]
    
    # 3. User Confirmation
    print(f"\nWARNING: You are about to permanently delete Node {node_id}.")
    print(f"Target Directory: {folder_path}")
    confirm = input("Type 'YES' to confirm deletion: ")
    
    if confirm == 'YES':
        # Delete from Disk
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"-> Deleted folder: {folder_path}")
        else:
            print(f"-> Directory not found on disk (already deleted?).")
            
        # Delete from DB
        cursor.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
        conn.commit()
        print(f"-> Removed Node {node_id} from SQLite database.")
    else:
        print("Pruning aborted.")
        
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FYP Lineage Tracker CLI")
    parser.add_argument('--action', choices=['init', 'register_root'], default='init')
    parser.add_argument('--origin', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--folder_path', type=str)
    parser.add_argument('--nickname', type=str)
    parser.add_argument('--history_log', type=str)
    parser.add_argument('--root_id', type=str)
    parser.add_argument('--timestamp', type=str)

    args = parser.parse_args()

    if args.action == 'init':
        init_db()
    elif args.action == 'register_root':
        register_root_dataset(
            args.origin, args.method, args.folder_path, 
            args.nickname, args.history_log, 
            args.root_id, args.timestamp
        )