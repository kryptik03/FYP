import os
import json

loc_nodes = ['xJDG', 'Rcij', 'FuKN', 'gFj0', 'IQoQ', 'VeKg', 'xgPI', 'Kdif', '4TDQ', 'dvAZ', '5IoP']
clf_nodes = ['vTgJ', 'zozf', 'CA4O', 'lPxq', 'wmrX', 'h8bC', '0D29', '47QD', 'HAVn', 'BNuP', 'r06G']
loc_dir = 'data/localisation_output'
clf_dirs = [
    'data/classification_output/exp06_dec', 
    'data/classification_output/exp07_supcon_dec', 
    'data/classification_output/exp08_supcon_dec_vit', 
    'data/classification_output/exp09_dec'
]

results = []

for i in range(len(loc_nodes)):
    lnode = loc_nodes[i]
    cnode = clf_nodes[i]
    
    loc_file = None
    if os.path.exists(loc_dir):
        for d in os.listdir(loc_dir):
            if lnode in d:
                loc_file = os.path.join(loc_dir, d, 'localisation_results.json')
                break
                
    clf_file = None
    for cdir in clf_dirs:
        if os.path.exists(cdir):
            for d in os.listdir(cdir):
                if cnode in d:
                    clf_file = os.path.join(cdir, d, 'metrics.json')
                    break
            if clf_file: break
            
    print(f'\n--- Node Pair: Loc({lnode}) / Clf({cnode}) ---')
    if clf_file and os.path.exists(clf_file):
        with open(clf_file, 'r') as f:
            c = json.load(f)
            print(f"Clf: Acc={c.get('classification_accuracy')} GrpRec={c.get('grouping_recall')} GrpPrec={c.get('grouping_precision')} Silh={c.get('silhouette_score')}")
    else:
        print('Clf json not found')
        
    if loc_file and os.path.exists(loc_file):
        with open(loc_file, 'r') as f:
            l = json.load(f)
            print('Loc Results:')
            for res in l:
                print(f"  Class {res.get('pred_class_id')} (GT {res.get('gt_class_id')}): ValidInst={res.get('num_valid_instances')} Err(m)={res.get('error_m')} Err(%)={res.get('error_pct')}")
    else:
        print('Loc json not found')
