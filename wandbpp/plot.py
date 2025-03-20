import logging
import re
from tqdm import tqdm
import yaml
import os
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any, Optional

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
        # Extract configuration parameters
        self.config = dict(config)
        self.runs_path = self.config.get("runs_path", "data/wandb_exports")
        self.file_scalars: str = self.config.get("file_scalars", "scalars.csv")
        self.file_config: str = self.config.get("file_config", "config.yaml")
        self.metric: str = self.config["metric"]
        self.filters: List[str] = self.config.get("filters", [])
        self.grouping_fields: List[str] = self.config.get("grouping", [])
        self.method_aggregate: str = self.config.get("method_aggregate", "mean")
        self.method_error: str = self.config.get("method_error", "std")
        self.max_n_runs: int = self.config.get("max_n_runs", np.inf)
        if self.max_n_runs in [None, "None", "null", "inf", "np.inf"]:
            self.max_n_runs = np.inf
        self.x_axis: str = self.config.get("x_axis", "_step")
        self.x_lim = self.config.get("x_lim", None)
        if self.x_lim is None:
            self.x_lim = [None, None]
        self.y_lim = self.config.get("y_lim", None)
        if self.y_lim is None:
            self.y_lim = [None, None]
        self.do_try_include_y0: bool = self.config.get("do_try_include_y0", False)
        self.ratio_near_y0: str = self.config.get("ratio_near_y0", 0.5)
        self.do_grid: bool = self.config.get("do_grid", False)

        # Define metric and grouping fields representation
        self.grouping_fields_repr = [
            field.split(".")[-1] for field in self.grouping_fields
        ]
        if len(self.grouping_fields_repr) == 0:
            self.grouping_fields_repr = ["All"]
        elif len(self.grouping_fields_repr) == 1:
            self.grouping_fields_repr = self.grouping_fields_repr[0]
        else:
            self.grouping_fields_repr = f"({', '.join(self.grouping_fields_repr)})"
        self.metric_repr = self.metric.replace("metrics.", "")

        # Define variables
        self.run_dirs: List[str] = []
        self.runs: List[Dict[str, Any]] = []
        self.grouped_data: Dict[Any, List[Dict[str, Any]]] = {}

    def load_run_data(self, run_path: str) -> Dict[str, Any]:
        """Loads scalar metrics and config from a run directory.

        Args:
            run_path (str): Path to the run directory.

        Returns:
            Dict[str, Any]: Dictionary containing scalars DataFrame and run config.
        """
        scalars_path = os.path.join(run_path, self.file_scalars)
        config_path = os.path.join(run_path, self.file_config)

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
                if not eval(filter_condition, {"config": run_config, "re": re}):
                    return False
            except Exception as e:
                print(f"Warning: Could not evaluate filter '{filter_condition}' - {e}")
                return False
        return True

    def load_and_filter_run_data(self):
        n_run = 0
        for run_dir in tqdm(self.run_dirs, desc="[wandbpp plot] Filtering ..."):
            if n_run >= self.max_n_runs:
                break
            run_data = self.load_run_data(run_dir)
            if run_data and self.apply_filters(run_data):
                self.runs.append(run_data)
                n_run += 1
        logger.info(f"[wandbpp plot] Loaded {len(self.runs)} runs after filtering.")

    def group_runs(self):
        """Groups runs based on specified grouping fields."""
        if len(self.grouping_fields) == 0:
            self.grouped_data = {("All",): self.runs}
            return
        logger.info(f"[wandbpp plot] Grouping runs by {self.grouping_fields_repr}...")
        self.grouped_data = {}
        for run in self.runs:
            run_config = run["config"]
            group_key = []
            for field in self.grouping_fields:
                try:
                    group_key.append(str(eval(field, {"config": run_config})))
                except Exception as e:
                    logger.warning(
                        f"[wandbpp plot warning] Field '{field}' could not be evaluated, assigning null instead. Error : {e}"
                    )
                    group_key.append("None")
            if len(group_key) == 1:
                group_key = group_key[0]
            else:
                group_key = ", ".join(group_key)

            if group_key not in self.grouped_data:
                self.grouped_data[group_key] = []
            self.grouped_data[group_key].append(run)
        grouped_data_keys = list(self.grouped_data.keys())
        lengths_groups = [len(v) for v in self.grouped_data.values()]
        logger.info(
            f"[wandbpp plot] Obtained {len(grouped_data_keys)} groups by grouping by {self.grouping_fields_repr} : {grouped_data_keys if len(grouped_data_keys) < 10 else grouped_data_keys[:10] + ['...']}, of size {lengths_groups if len(lengths_groups) < 10 else lengths_groups[:10] + ['...']} (average {np.mean(lengths_groups):.2f} ± {np.std(lengths_groups):.2f}, runs/group, min {np.min(lengths_groups)}, max {np.max(lengths_groups)})"
        )

    def try_include_y0(
        self, y_lim: Optional[List[float]], y_min: float, y_max: float
    ) -> List[float]:
        """Tries to include y=0 in the y-axis limits if 0 is close to the points relatively to the range |y_max - y_min|.

        Args:
            y_lim (Optional[List[float]]): Y-axis limits.
            merged_df (pd.DataFrame): DataFrame containing all metric values.

        Returns:
            List[float]: Updated y-axis limits.
        """
        if y_min <= 0 <= y_max:  # negative and positive values (already includes 0)
            return y_lim
        range_values = y_max - y_min
        if y_min <= y_max <= 0:  # negative values only
            if range_values > self.ratio_near_y0 * -y_min and y_lim[1] is None:
                return [y_min, 0]
            else:
                return y_lim
        elif 0 <= y_min <= y_max:  # positive values only
            if range_values > self.ratio_near_y0 * y_max and y_lim[0] is None:
                return [0, y_max]
            else:
                return y_lim
        else:
            raise ValueError("Invalid y-axis limits.")

    def plot_grouped_data(self):
        """Plots metrics with mean and standard error for each group.
        Also defines y_min and y_max based on the metric values.
        """
        plt.figure(figsize=(10, 6))
        self.y_min, self.y_max = np.inf, -np.inf

        if (
            len(self.grouped_data) == 1
        ):  # If only one group, use the method_aggregate as the key
            self.grouped_data = {
                self.method_aggregate: v for k, v in self.grouped_data.items()
            }

        for group_key, runs in self.grouped_data.items():
            # Get the merged DataFrame for all runs in the group
            list_group_dfs: List[pd.DataFrame] = []
            df_x_values = pd.DataFrame()
            max_t = -np.inf
            for run in runs:
                df: pd.DataFrame = run["scalars"]
                if self.x_axis in df.columns:
                    try:
                        metric_values = eval(self.metric, {"metrics": df, "np": np})
                        df[self.metric] = metric_values
                        list_group_dfs.append(df[[self.metric]])
                        # Get the x-axis values (take last DataFrame as reference)
                        if max_t < df[self.x_axis].max():
                            max_t = df[self.x_axis].max()
                            df_x_values = df[[self.x_axis]]
                    except Exception as e:
                        logger.warning(
                            f"[wandbpp plot warning] Could not evaluate metric expression '{self.metric}' - {e}"
                        )
                        continue
                else:
                    logger.warning(
                        f"[wandbpp plot warning] Could not find x-axis column '{self.x_axis}' in DataFrame for run {run['config']['name']}"
                    )
            if not list_group_dfs:  # Skip group if no valid data
                logger.warning(
                    f"[wandbpp plot warning] Skipping group {group_key} due to missing data."
                )
                continue
            merged_df = pd.concat(list_group_dfs, axis=1, join="outer")
            x_values = df_x_values[self.x_axis]

            # Compute aggregated values
            mean_values = merged_df.mean(axis=1, skipna=True)
            if self.method_aggregate == "mean":
                values_aggregated = mean_values
            elif self.method_aggregate == "median":
                values_aggregated = merged_df.median(axis=1, skipna=True)
            elif self.method_aggregate == "max":
                values_aggregated = merged_df.max(axis=1, skipna=True)
            elif self.method_aggregate == "min":
                values_aggregated = merged_df.min(axis=1, skipna=True)
            else:
                raise ValueError(f"Invalid method_aggregate: {self.method_aggregate}")

            # Compute error values
            if self.method_error == "std":
                delta_low = delta_high = merged_df.std(axis=1, skipna=True)
            elif self.method_error == "sem":
                delta_low = delta_high = merged_df.sem(axis=1, skipna=True)
            elif self.method_error == "range":
                delta_low = mean_values - merged_df.min(axis=1, skipna=True)
                delta_high = merged_df.max(axis=1, skipna=True) - mean_values
            elif self.method_error == "none":
                delta_low = delta_high = None
            else:
                raise ValueError(f"Invalid method_error: {self.method_error}")

            # Plot aggregated values
            plt.plot(x_values, values_aggregated, label=f"{group_key}")
            if delta_low is not None and delta_high is not None:
                plt.fill_between(
                    x_values,
                    mean_values - delta_low,
                    mean_values + delta_high,
                    alpha=0.2,
                )

            # Update y_min and y_max based on the group values
            if delta_low is not None and delta_high is not None:
                self.y_min = min(values_aggregated.min() - delta_low.min(), self.y_min)
                self.y_max = max(values_aggregated.max() + delta_high.max(), self.y_max)
            else:
                self.y_min = min(values_aggregated.min(), self.y_min)
                self.y_max = max(values_aggregated.max(), self.y_max)

        # Don't plot if no data
        if self.y_min == np.inf or self.y_max == -np.inf:
            logger.warning("[wandbpp plot warning] No data to plot.")
            return

        # Plot settings
        plt.xlabel("Steps")
        plt.ylabel(self.metric_repr)
        plt.xlim(self.x_lim)
        if self.do_try_include_y0:
            self.y_lim = self.try_include_y0(
                self.y_lim, y_min=self.y_min, y_max=self.y_max
            )
        plt.ylim(self.y_lim)
        plt.legend()
        plt.grid(visible=self.config["kwargs_grid"])
        if self.method_error == "none":
            string_methods_agg_error = f"{self.method_aggregate}"
        else:
            string_methods_agg_error = f"{self.method_aggregate}, σ={self.method_error}"
        if len(self.grouping_fields) == 0:
            title = f"{self.metric_repr} ({string_methods_agg_error})"
        else:
            title = f"{self.metric_repr} (grouped by {self.grouping_fields_repr} : {string_methods_agg_error})"
        plt.title(title)

        plt.show()

    def run(self):
        """Executes the full pipeline: loading, filtering, grouping, and plotting."""
        self.run_dirs = [
            os.path.join(self.runs_path, d)
            for d in os.listdir(self.runs_path)
            if os.path.isdir(os.path.join(self.runs_path, d))
        ]
        logger.info(
            f"[wandbpp plot] Found {len(self.run_dirs)} runs in {self.runs_path}."
        )

        # Load and filter run data
        self.load_and_filter_run_data()
        if len(self.runs) == 0:
            logger.warning("[wandbpp plot warning] No runs to plot.")
            return

        # Group runs based on specified fields
        self.group_runs()
        if sum(len(v) for v in self.grouped_data.values()) == 0:
            raise ValueError(
                "[wandbpp plot] Run were loaded but no runs were grouped. This should not happen."
            )
            logger.warning("[wandbpp plot] No groups to plot.")
            return

        # Plot grouped data
        self.plot_grouped_data()
        logger.info("[wandbpp plot] Plotting complete.")


@hydra.main(config_path="../configs_plot", config_name="config_default.yaml")
def main(config: DictConfig):
    """Main function to initialize and run the Plotter."""
    plotter = Plotter(config)
    plotter.run()


if __name__ == "__main__":
    main()
