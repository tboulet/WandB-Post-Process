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


def parse_filters(config: Dict[str, Any]) -> List[str]:
    """Extracts filter conditions from the config."""
    return config.get("filters", [])


def parse_grouping(config: Dict[str, Any]) -> List[str]:
    """Extracts grouping fields from the config."""
    return config.get("grouping", [])


def load_run_data(run_path: str) -> Dict[str, Any]:
    """Loads scalar metrics and config from a run directory."""
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
    """Applies filters to determine if a run should be included."""
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
    """Groups runs based on specified grouping fields."""
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


def plot_grouped_data(grouped_data: Dict[tuple, List[Dict[str, Any]]]):
    """Plots metrics with mean and standard error for each group."""
    plt.figure(figsize=(10, 6))

    for group_key, runs in grouped_data.items():
        all_steps = []
        all_values = []

        for run in runs:
            df = run["scalars"]
            if "_step" in df.columns:
                steps = df["_step"].values
                metric_name = [col for col in df.columns if col != "_step"][
                    0
                ]  # Assuming one metric
                values = df[metric_name].values

                all_steps.append(steps)
                all_values.append(values)

        mean_values = np.mean(all_values, axis=0)
        std_error = np.std(all_values, axis=0) / np.sqrt(len(runs))
        plt.plot(steps, mean_values, label=f"Group: {group_key}")
        plt.fill_between(
            steps, mean_values - std_error, mean_values + std_error, alpha=0.2
        )

    plt.xlabel("Steps")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Aggregated Metrics by Group")
    plt.show()


@hydra.main(config_path="../configs_plot", config_name="config_default.yaml")
def main(config: DictConfig):
    config = dict(config)  # Convert to standard dict if needed
    filters = config.get("filters", [])
    grouping_fields = config.get("grouping", [])
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

    plot_grouped_data(grouped_data)
    logger.info("")


if __name__ == "__main__":
    main()
