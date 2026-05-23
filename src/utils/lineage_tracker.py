import sqlite3
import os
import shutil
from datetime import datetime
import string
import random

# lineage.db lives alongside this script in src/utils/ so that git tracks it.
# Colab can push the updated DB back to GitHub after each training run.
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lineage.db'))

def init_db():
    """Initializes the SQLite database and the Edge List table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH) # connects sqlite database, creating one if none exists.
    # stores the connection object in 'conn'

    cursor = conn.cursor() # create a cursor object for the connection. 
    
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
    conn.commit() # commits the changes to the database
    conn.close() # closes connection
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
    # Normalize to absolute path so pruning works regardless of CWD
    folder_path = os.path.abspath(folder_path)
    
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

def register_process(parent_id, stage, method, folder_path, appended_history, force_node_id=None, force_timestamp=None):
    """Registers a downstream process (Child Node) linked to a parent."""
    node_id = force_node_id if force_node_id else generate_short_id()
    timestamp = force_timestamp if force_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    # Normalize to absolute path so pruning works regardless of CWD
    folder_path = os.path.abspath(folder_path)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch parent details to inherit origin, root_id, nickname, and previous history
    cursor.execute("SELECT root_id, origin, nickname, history_log FROM nodes WHERE node_id=?", (parent_id,))
    parent_data = cursor.fetchone() # returns the fetched row as a python tuple. If no row is found, it returns None. 
    
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

def get_node_history(node_id: str) -> str:
    """Returns the full history log of a node as a string."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT history_log FROM nodes WHERE node_id=?", (node_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    else:
        raise ValueError(f"Node {node_id} does not exist in the database.")


def describe_node(node_id):
    """Displays a highly detailed hierarchical database metadata report for the selected node and all its descendants (leaves)."""
    if not os.path.exists(DB_PATH):
        print("[Error] Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT node_id, parent_id, root_id, origin, stage, method, folder_path, nickname, timestamp, history_log 
        FROM nodes
    """)
    all_rows = cursor.fetchall()
    conn.close()

    # Build node maps and parent-child adjacency list
    nodes_by_id = {row[0]: row for row in all_rows}
    if node_id not in nodes_by_id:
        print(f"[Error] Node '{node_id}' does not exist in the lineage database.")
        return

    from collections import defaultdict
    children_map = defaultdict(list)
    for row in all_rows:
        pid = row[1]
        nid = row[0]
        children_map[pid].append(nid)

    # Perform a Depth-First Search (DFS) starting from the selected node to get descendants with depth
    traversal_order = []
    def dfs(current_id, depth=0):
        traversal_order.append((current_id, depth))
        for child_id in children_map[current_id]:
            dfs(child_id, depth + 1)

    dfs(node_id)

    print("\n" + "="*80)
    print(f" LINEAGE METADATA REPORT: Hierarchy of '{node_id}' ({len(traversal_order)} node(s)) ".center(80, "="))
    print("="*80)

    for nid, depth in traversal_order:
        row = nodes_by_id[nid]
        _, pid, rid, origin, stage, method, path, nickname, ts, history = row
        
        indent = "    " * depth
        prefix = f"ROOT NODE: {nid}" if depth == 0 else f"{indent}+-- CHILD NODE: {nid}"
        
        print(f"\n{prefix}")
        print(f"{indent}  -------------------------------------------------------------")
        print(f"{indent}  Node ID     : {nid}")
        print(f"{indent}  Parent ID   : {pid}")
        print(f"{indent}  Root ID     : {rid}")
        print(f"{indent}  Origin      : {origin.upper()}")
        print(f"{indent}  Stage       : {stage.upper()}")
        print(f"{indent}  Method      : {method}")
        print(f"{indent}  Nickname    : '{nickname}'")
        print(f"{indent}  Timestamp   : {ts}")
        print(f"{indent}  Folder Path : {path}")
        print(f"{indent}  History Log : {history}")
        print(f"{indent}  -------------------------------------------------------------")
        
    print("="*80 + "\n")

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
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"-> Deleted folder: {folder_path}")
            else:
                os.remove(folder_path)
                print(f"-> Deleted file: {folder_path}")
        else:
            print(f"-> Path not found on disk (already deleted?).")
            
        # Delete from DB
        cursor.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
        conn.commit()
        print(f"-> Removed Node {node_id} from SQLite database.")

def list_roots():
    """Lists all nodes that have no parent (Root Datasets)."""
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT node_id, stage, method, nickname, timestamp 
        FROM nodes 
        WHERE parent_id = 'NONE'
        ORDER BY timestamp DESC
    """)
    roots = cursor.fetchall()
    
    if not roots:
        print("No root datasets found.")
    else:
        print("\n=== Registered Root Datasets ===")
        print(f"{'ID':<6} | {'Stage':<12} | {'Method':<18} | {'Nickname'}")
        print("-" * 60)
        for r in roots:
            print(f"{r[0]:<6} | {r[1]:<12} | {r[2]:<18} | {r[3]}")
        print("-" * 60)
    
    conn.close()


def visualize_tree(root_id):
    """Builds and prints a visual DAG tree in the terminal."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT node_id, parent_id, stage, method, nickname FROM nodes WHERE root_id=?", (root_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"[Error] No data found for Root ID: {root_id}")
        return

    from collections import defaultdict
    tree = defaultdict(list)
    '''
    When you write tree = defaultdict(list), you are telling Python: "Make a dictionary. 
    If I ever try to access a key that doesn't exist, do not throw an error. 
    Instead, automatically create that key and set its default value to an empty list
    '''

    nodes_info = {}
    
    for row in rows:
        nid, pid, stage, method, nickname = row
        tree[pid].append(nid) # Add the child ID (nid) to the parent's (pid) list
        nodes_info[nid] = {'stage': stage, 'method': method, 'nickname': nickname}

    print(f"\n=== Lineage Tree for Root: {root_id} ===")

    def print_node(node, prefix="", is_last=True):
        info = nodes_info.get(node, {})
        stage_method = f"[{info.get('stage')} : {info.get('method')}]"

        # Print the current node
        if node == root_id:
            print(f"[ROOT] {node} {stage_method} ('{info.get('nickname')}')")
        else:
            branch = "+-- " if is_last else "+-- "
            print(f"{prefix}{branch}[NODE] {node} {stage_method}")

        # Recursively print children
        children = tree.get(node, [])
        for i, child in enumerate(children):
            next_is_last = (i == len(children) - 1)
            # Root node doesn't add a prefix indent to its immediate children
            next_prefix = prefix + ("    " if is_last else "|   ") if node != root_id else ""
            print_node(child, next_prefix, next_is_last)
        # dayum this's a really smart way of doing it!

    print_node(root_id)
    print("==================================\n")


def visualize_all():
    """Fetches all root datasets and visualizes the tree structure for each of them."""
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT node_id, nickname 
        FROM nodes 
        WHERE parent_id = 'NONE'
        ORDER BY timestamp DESC
    """)
    roots = cursor.fetchall()
    conn.close()

    if not roots:
        print("No root datasets found to visualize.")
        return

    print(f"\n=== Visualizing All Lineage Trees ({len(roots)} root(s) found) ===")
    for r in roots:
        root_id, nickname = r
        visualize_tree(root_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FYP Lineage Tracker CLI")
    
    # NEW: Added 'all' and 'inspect' to the choices
    parser.add_argument('--action', choices=['init', 'register_root', 'visualize', 'prune', 'register_process', 'roots', 'all', 'inspect'], default='init')
    
    parser.add_argument('--origin', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--folder_path', type=str)
    parser.add_argument('--nickname', type=str)
    parser.add_argument('--history_log', type=str)
    parser.add_argument('--root_id', type=str)
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--node_id', type=str) # NEW: For pruning
    parser.add_argument('--parent_id', type=str) # NEW: For registering a process
    parser.add_argument('--stage', type=str) # NEW: For registering a process

    args = parser.parse_args()

    if args.action == 'init':
        init_db()
    elif args.action == 'register_root':
        register_root_dataset(
            args.origin, args.method, args.folder_path, 
            args.nickname, args.history_log, 
            args.root_id, args.timestamp
        )
    elif args.action == 'visualize':
        if not args.root_id:
            print("Error: --root_id is required for visualization.")
        else:
            visualize_tree(args.root_id)
    elif args.action == 'register_process':
        register_process(
            args.parent_id, args.stage, args.method, args.folder_path, 
            args.history_log, args.node_id, args.timestamp
        )
    elif args.action == 'prune':
        if not args.node_id:
            print("Error: --node_id is required for pruning.")
        else:
            prune_node(args.node_id)
    elif args.action == 'roots':
        list_roots()
    elif args.action == 'all':
        visualize_all()
    elif args.action == 'inspect':
        if not args.node_id:
            print("Error: --node_id is required for inspection.")
        else:
            describe_node(args.node_id)