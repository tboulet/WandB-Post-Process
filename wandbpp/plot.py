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


class Plotter:
    def __init__(self, config: DictConfig):
        """Initializes the Plotter class with configuration parameters.

        Args:
            config (DictConfig): Hydra configuration object.
        """
        self.config = dict(config)  # Convert to standard dict if needed
        self.filters: List[str] = self.config.get("filters", [])
        self.grouping_fields: List[str] = self.config.get("grouping", [])
        self.metric: str = self.config["metric"]
        self.runs_path = self.config.get("runs_path", "data/wandb_exports")
        self.grouping_fields_repr = ", ".join(
            [field.split(".")[-1] for field in self.grouping_fields]
        )
        self.runs = []
        self.grouped_data = {}
        self.x_lim = self.config.get("x_lim", None)
        self.y_lim = self.config.get("y_lim", None)

    def load_run_data(self, run_path: str) -> Dict[str, Any]:
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

    def apply_filters(self, run_data: Dict[str, Any]) -> bool:
        """Applies filters to determine if a run should be included.

        Args:
            run_data (Dict[str, Any]): Run data containing scalars and config.

        Returns:
            bool: True if run satisfies all filters, False otherwise.
        """
        run_config = run_data["config"]
        for filter_condition in self.filters:
            try:
                if not eval(filter_condition, {"config": run_config}):
                    return False
            except Exception as e:
                print(f"Warning: Could not evaluate filter '{filter_condition}' - {e}")
                return False
        return True

    def group_runs(self):
        """Groups runs based on specified grouping fields."""
        self.grouped_data = {}
        for run in self.runs:
            run_config = run["config"]
            try:
                group_key = tuple(
                    eval(field, {"config": run_config})
                    for field in self.grouping_fields
                )
                if len(group_key) == 1:
                    group_key = group_key[0]
            except Exception as e:
                logger.warning(
                    f"Could not evaluate grouping field '{self.grouping_fields}' - {e}"
                )
                group_key = tuple("Unknown" for _ in self.grouping_fields)

            if group_key not in self.grouped_data:
                self.grouped_data[group_key] = []
            self.grouped_data[group_key].append(run)

    def plot_grouped_data(self):
        """Plots metrics with mean and standard error for each group."""
        plt.figure(figsize=(10, 6))

        for group_key, runs in self.grouped_data.items():
            all_dfs = []

            for run in runs:
                df = run["scalars"]
                if "_step" in df.columns and self.metric in df.columns:
                    df = df.set_index("_step")[[self.metric]]
                    all_dfs.append(df)

            if not all_dfs:
                continue

            merged_df = pd.concat(all_dfs, axis=1, join="outer")
            mean_values = merged_df.mean(axis=1, skipna=True)
            std_error = merged_df.std(axis=1, skipna=True) / np.sqrt(len(runs))

            plt.plot(mean_values.index, mean_values, label=f"{group_key}")
            plt.fill_between(
                mean_values.index,
                mean_values - std_error,
                mean_values + std_error,
                alpha=0.2,
            )

        plt.xlabel("Steps")
        plt.ylabel(self.metric)
        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)
        plt.legend()
        plt.title(f"{self.metric} aggregated by {self.grouping_fields_repr}")

        plt.show()

    def run(self):
        """Executes the full pipeline: loading, filtering, grouping, and plotting."""
        run_dirs = [
            os.path.join(self.runs_path, d)
            for d in os.listdir(self.runs_path)
            if os.path.isdir(os.path.join(self.runs_path, d))
        ]
        logger.info(f"Found {len(run_dirs)} runs in {self.runs_path}.")

        for run_dir in tqdm(run_dirs, desc="Filtering ..."):
            run_data = self.load_run_data(run_dir)
            if run_data and self.apply_filters(run_data):
                self.runs.append(run_data)
        logger.info(f"Loaded {len(self.runs)} runs after filtering.")

        logger.info(f"Grouping runs by {self.grouping_fields_repr}...")
        self.group_runs()
        grouped_data_keys = list(self.grouped_data.keys())
        logger.info(
            f"Obtained {len(grouped_data_keys)} groups by grouping by {self.grouping_fields_repr} to obtain {grouped_data_keys if len(grouped_data_keys) < 10 else grouped_data_keys[:10] + ['...']}"
        )

        self.plot_grouped_data()
        logger.info("Plotting complete.")


@hydra.main(config_path="../configs_plot", config_name="config_default.yaml")
def main(config: DictConfig):
    """Main function to initialize and run the Plotter."""
    plotter = Plotter(config)
    plotter.run()


if __name__ == "__main__":
    main()
