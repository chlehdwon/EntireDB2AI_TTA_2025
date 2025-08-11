import os
os.environ['XDG_CACHE_HOME'] = '/data/cache'

from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

dataset = "rel-stack"
task_idx_dict = {"user-engagement": {"best_idx": 1914, "single_table": 63.39}, "user-badge": {"best_idx": 1281, "single_table": 63.43}}

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
            save_csv=False) * 100

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset} | Task: {task}")
    print(f"{'='*50}")
    
    # Calculate improvement
    improvement = (metric / single_table_perf - 1) * 100
    
    # Print table with nice formatting using string formatting
    print("\n[Performance Comparison]")
    print(f"{'Model':<15} {'AUC-ROC (%)':<12} {'Improvement (%)':<15}")
    print("-" * 45)
    print(f"{'Single-Table':<15} {single_table_perf:<12.2f} {'-':<15}")
    print(f"{'EntireDB2AI':<15} {metric:<12.2f} {improvement:<15.2f}")
    print(f"{'='*50}\n")
