import logging
from tqdm import tqdm
import yaml
import os
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_run_data(run_path: str) -> Dict[str, Any]:
    """Loads scalar metrics and config from a run directory.
    
    Args:
        run_path (str): Path to the run directory.
    
    Returns:
        Dict[str, Any]: Dictionary containing scalars DataFrame and run config.
    """
    scalars_path = os.path.join(run_path, "scalars.csv")
    config_path = os.path.join(run_path, "config.yaml")

    if not os.path.exists(scalars_path) or not os.path.exists(config_path):
        return None

    scalars_df = pd.read_csv(scalars_path)
    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f)
    run_config = OmegaConf.create(run_config)
    return {"scalars": scalars_df, "config": run_config}

def apply_filters(run_data: Dict[str, Any], filters: List[str]) -> bool:
    """Applies filters to determine if a run should be included.
    
    Args:
        run_data (Dict[str, Any]): Run data containing scalars and config.
        filters (List[str]): List of filter conditions.
    
    Returns:
        bool: True if run satisfies all filters, False otherwise.
    """
    run_config = run_data["config"]
    for filter_condition in filters:
        try:
            if not eval(filter_condition, {"config": run_config}):
                return False
        except Exception as e:
            print(f"Warning: Could not evaluate filter '{filter_condition}' - {e}")
            return False
    return True

def group_runs(
    runs: List[Dict[str, Any]], grouping_fields: List[str]
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Groups runs based on specified grouping fields.
    
    Args:
        runs (List[Dict[str, Any]]): List of run data.
        grouping_fields (List[str]): List of fields to group by.
    
    Returns:
        Dict[tuple, List[Dict[str, Any]]]: Dictionary mapping group keys to runs.
    """
    grouped_data = {}
    for run in runs:
        run_config = run["config"]
        try:
            group_key = tuple(
                eval(field, {"config": run_config}) for field in grouping_fields
            )
        except Exception as e:
            logger.warning(
                f"Could not evaluate grouping field '{grouping_fields}' - {e}"
            )
            group_key = tuple("Unknown" for _ in grouping_fields)

        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(run)

    return grouped_data

def plot_grouped_data(grouped_data: Dict[tuple, List[Dict[str, Any]]], metric: str):
    """Plots metrics with mean and standard error for each group.
    
    Args:
        grouped_data (Dict[tuple, List[Dict[str, Any]]]): Grouped run data.
        metric (str): Metric to plot.
    """
    plt.figure(figsize=(10, 6))

    for group_key, runs in grouped_data.items():
        all_dfs = []

        for run in runs:
            df = run["scalars"]
            if "_step" in df.columns and metric in df.columns:
                df = df.set_index("_step")[[metric]]
                all_dfs.append(df)

        if not all_dfs:
            continue

        merged_df = pd.concat(all_dfs, axis=1, join="outer")
        mean_values = merged_df.mean(axis=1, skipna=True)
        std_error = merged_df.std(axis=1, skipna=True) / np.sqrt(len(runs))

        plt.plot(mean_values.index, mean_values, label=f"Group: {group_key}")
        plt.fill_between(
            mean_values.index, mean_values - std_error, mean_values + std_error, alpha=0.2
        )

    plt.xlabel("Steps")
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"{metric} by Group")
    plt.show()

@hydra.main(config_path="../configs_plot", config_name="config_default.yaml")
def main(config: DictConfig):
    """Main function for loading, filtering, grouping, and plotting run data.
    
    Args:
        config (DictConfig): Hydra configuration object.
    """
    config = dict(config)  # Convert to standard dict if needed
    filters = config.get("filters", [])
    grouping_fields = config.get("grouping", [])
    metric = config.get("metric", "loss")  # Default metric to "loss"
    runs_path = config.get("runs_path", "data/wandb_exports")

    run_dirs = [
        os.path.join(runs_path, d)
        for d in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, d))
    ]

    runs = []
    for run_dir in tqdm(
        run_dirs, desc=f"Loading runs from {runs_path} (total: {len(run_dirs)})..."
    ):
        run_data = load_run_data(run_dir)
        if run_data and apply_filters(run_data, filters):
            runs.append(run_data)
    logger.info(f"Loaded {len(runs)} runs after filtering.")

    grouped_data = group_runs(runs, grouping_fields)
    grouped_data_keys = list(grouped_data.keys())
    logger.info(
        f"Grouped data by fields {grouping_fields} to obtain {grouped_data_keys if len(grouped_data_keys) < 10 else grouped_data_keys[:10] + ['...']} (total: {len(grouped_data_keys)})"
    )

    plot_grouped_data(grouped_data, metric)
    logger.info("")

if __name__ == "__main__":
    main()