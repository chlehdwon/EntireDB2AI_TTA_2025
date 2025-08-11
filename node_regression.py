import os
os.environ['XDG_CACHE_HOME'] = '/data/cache'

from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

dataset = "rel-amazon"
task_idx_dict = {"item-ltv": {"best_idx": 0, "single_table": 60.569}}

for task, info_dict in task_idx_dict.items():
    best_idx, single_table_perf = info_dict["best_idx"], info_dict["single_table"]
    metric = run_gnn_node_worker(
            dataset_name=dataset,
            task_name=task,
            gnn="GraphSAGE",
            cache_dir="/data/starlab/relbench_examples",
            tag="save",
            target_indices=[best_idx],
            model_load_path=f"./checkpoints/{dataset}_{task}.pt",
            save_csv=False)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset} | Task: {task}")
    print(f"{'='*50}")
    
    # Calculate improvement
    improvement = (single_table_perf / metric - 1) * 100
    
    # Print table with nice formatting using string formatting
    print("\n[Performance Comparison]")
    print(f"{'Model':<15} {'MAE':<12} {'Improvement (%)':<15}")
    print("-" * 45)
    print(f"{'Single-Table':<15} {single_table_perf:<12.3f} {'-':<15}")
    print(f"{'EntireDB2AI':<15} {metric:<12.3f} {improvement:<15.2f}")
    print(f"{'='*50}\n")