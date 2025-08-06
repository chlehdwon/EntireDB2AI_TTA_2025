import os
os.environ['XDG_CACHE_HOME'] = '/data/cache'

from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

dataset = "rel-hm"
task_idx_dict = {"item-sales": {"best_idx": 0, "single_table": 0.076}}

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
    print(f"Single-Table performance (AUC-ROC (%)) = {metric:.2f}")
    print(f"Our performance (AUC-ROC (%)) = {single_table_perf}")
    print(f"Improvement = {(single_table_perf - metric) / single_table_perf * 100:.2f}%")