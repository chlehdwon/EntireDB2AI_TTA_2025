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
            cache_dir="/data/cache/relbench_dataset",
            tag="save",
            target_indices=[best_idx],
            model_load_path=f"./checkpoints/{dataset}_{task}.pt",
            save_csv=False)
    print(f"Single-Table performance (AUC-ROC (%)) = {metric * 100:.2f}")
    print(f"Our performance (AUC-ROC (%)) = {single_table_perf}")
    print(f"Improvement = {(metric*100 - single_table_perf) / single_table_perf * 100:.2f}%")
