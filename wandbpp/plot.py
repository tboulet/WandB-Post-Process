import logging
from logging import basicConfig, getLogger
import re
import omegaconf
from tqdm import tqdm
import yaml
import os
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, config: DictConfig):
        """Initializes the Plotter class with configuration parameters.

        Args:
            config (DictConfig): Hydra configuration object.
        """
        # Data selection options
        self.config = dict(config)
        self.runs_path = self.config.get("runs_path", "data/wandb_exports")
        self.file_scalars: str = self.config.get("file_scalars", "scalars.csv")
        self.file_config: str = self.config.get("file_config", "config.yaml")
        self.metric: str = self.config["metric"]
        self.metric_repr = self.metric.replace("metrics.", "")
        self.filters: List[str] = self.config.get("filters", [])
        self.grouping_fields: List[str] = self.config.get("grouping", [])
        self.grouping_fields_repr = [
            str(field).split(".")[-1] for field in self.grouping_fields
        ]
        if len(self.grouping_fields) == 0:
            self.grouping_fields_repr = ["None"]
        elif len(self.grouping_fields) == 1:
            self.grouping_fields_repr = self.grouping_fields[0].split(".")[-1]
        else:
            self.grouping_fields_repr = [
                str(field).split(".")[-1] for field in self.grouping_fields
            ]
            self.grouping_fields_repr = ", ".join(self.grouping_fields_repr)
            self.grouping_fields_repr = f"({self.grouping_fields_repr})"
        # Plotting options
        self.method_aggregate: str = self.config.get("method_aggregate", "mean")
        assert self.method_aggregate in [
            "mean",
            "median",
            "max",
            "min",
        ], f"Invalid method_aggregate: {self.method_aggregate}"
        self.method_error: str = self.config.get("method_error", "std")
        assert self.method_error in [
            "std",
            "sem",
            "range",
            "iqr",
            "none",
        ] or self.method_error.startswith(
            "percentile"
        ), f"Invalid method_error: {self.method_error}"
        self.n_samples_shown: int = self.config.get("n_samples_shown", 0)
        self.max_n_runs: int = self.config.get("max_n_runs", np.inf)
        if self.max_n_runs in [None, "None", "null", "inf", "np.inf"]:
            self.max_n_runs = np.inf
        # Plotting interface options
        self.label_individual: str = self.config.get("label_individual", "run_name")
        self.label_group: str = self.config.get("label_group", "group_key")
        self.x_axis: str = self.config.get("x_axis", "_step")
        self.x_lim = self.config.get("x_lim", None)
        if self.x_lim is None:
            self.x_lim = [None, None]
        self.y_lim = self.config.get("y_lim", None)
        if self.y_lim is None:
            self.y_lim = [None, None]
        self.do_try_include_y0: bool = self.config.get("do_try_include_y0", False)
        self.ratio_near_y0: str = self.config.get("ratio_near_y0", 0.5)
        self.kwargs_grid: Dict[str, Any] = self.config.get("kwargs_grid", {})
        self.alpha_shaded: float = self.config.get("alpha_shaded", 0.2)
        self.max_legend_length: int = self.config.get("max_legend_length", 10)
        self.groups_plot_kwargs: List[Dict[str, Any]] = self.config.get(
            "groups_plot_kwargs", {}
        )
        self.default_plot_kwargs: Dict = self.config.get("default_plot_kwargs", {})

        # Define variables
        self.run_dirs: List[str] = []
        self.runs_data: List[Dict[str, Any]] = []
        self.grouped_data: Dict[Any, List[Dict[str, Any]]] = {}

        # Set logger
        basicConfig(
            level=logging.INFO,
            format="[WandbPP plot] %(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )

    def load_run_data(self, run_path: str) -> Dict[str, Any]:
        """Loads scalar metrics and config from a run directory.

        Args:
            run_path (str): Path to the run directory.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - scalars: DataFrame containing scalar metrics.
                - config: OmegaConf object containing the run configuration.
                - name: Name of the run.
        """
        # Load scalars
        scalars_path = os.path.join(run_path, self.file_scalars)
        if not os.path.exists(scalars_path):
            logger.error(
                f"Could not find scalars file {self.file_scalars} in {run_path}. Skipping run."
            )
            return None
        scalars_df = pd.read_csv(scalars_path)
        # Load config
        config_path = os.path.join(run_path, self.file_config)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                run_config = yaml.safe_load(f)
        else:
            logger.warning(
                f"Could not find config file {self.file_config} in {run_path}. Using empty config."
            )
            run_config = {}
        run_config = OmegaConf.create(run_config)
        # Get run name from run_path
        run_name = os.path.basename(run_path)
        # Return run data
        return {"scalars": scalars_df, "config": run_config, "name": run_name}

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
        """Loads run data and applies filters to select runs for plotting."""
        n_run = 0
        for run_dir in tqdm(self.run_dirs, desc="[WandbPP plot] Filtering ..."):
            if n_run >= self.max_n_runs:
                break
            run_data = self.load_run_data(run_dir)
            if run_data and self.apply_filters(run_data):
                self.runs_data.append(run_data)
                n_run += 1
        logger.info(f"Loaded {len(self.runs_data)} runs after filtering.")

    def group_runs(self):
        """Groups runs based on specified grouping fields."""
        # If no grouping fields, group all runs together
        if len(self.grouping_fields) == 0:
            self.grouped_data = self.grouped_data = {
                run["name"]: [run] for run in self.runs_data
            }
            self.grouped_plot_kwargs: Dict[Any, Dict[str, Any]] = {}
            return
        # Initialize mappings from field to plot kwargs
        self.field_to_plot_kwargs: Dict[str, Dict[str, Any]] = {}
        for idx, field in enumerate(self.grouping_fields):
            if idx >= len(self.groups_plot_kwargs):
                break  # don't try to get plot kwargs if not enough are provided
            self.field_to_plot_kwargs[field] = omegaconf.OmegaConf.to_container(
                self.groups_plot_kwargs[idx], resolve=True
            )
            self.field_to_plot_kwargs[field]["values"] = []
        # Initialize mappings from group key to runs and plot kwargs
        logger.info(f"Grouping runs by {self.grouping_fields_repr}...")
        self.grouped_data: Dict[Any, List[Dict[str, Any]]] = {}
        self.grouped_plot_kwargs: Dict[Any, Dict[str, Any]] = {}
        # Iterate on the runs data
        for run_data in self.runs_data:
            run_config = run_data["config"]
            run_name = run_data["name"]
            # Construct group key
            group_key = []
            for field in self.grouping_fields:
                try:
                    if field is None:  # null field means a dummy group
                        group_key.append("None")
                    else:
                        value = str(eval(field, {"config": run_config}))
                except Exception as e:
                    logger.warning(
                        f"Field '{field}' could not be evaluated for run {run_name}, assigning 'None' instead - {e}"
                    )
                    value = "None"  # assign 'None' if field evaluation fails
                group_key.append(value)
            group_key = tuple(group_key)  # use tuple for hashability
            # Add eventually new value for field to plot kwargs
            for field, value in zip(self.grouping_fields, group_key):
                if (
                    field in self.field_to_plot_kwargs
                    and value not in self.field_to_plot_kwargs[field]["values"]
                ):
                    self.field_to_plot_kwargs[field]["values"].append(value)
            # If new group key, initialize list of runs as empty, and extract plot kwargs from field_to_plot_kwargs
            if group_key not in self.grouped_data:
                self.grouped_data[group_key] = []
                plot_kwargs = {}
                for field, value in zip(self.grouping_fields, group_key):
                    if field in self.field_to_plot_kwargs:
                        key = self.field_to_plot_kwargs[field]["key"]
                        values = self.field_to_plot_kwargs[field]["values"]
                        idx = values.index(value) % len(
                            self.field_to_plot_kwargs[field]["args"]
                        )
                        arg = self.field_to_plot_kwargs[field]["args"][idx]
                        plot_kwargs[key] = arg
                self.grouped_plot_kwargs[group_key] = plot_kwargs
            # Append run data to group
            self.grouped_data[group_key].append(run_data)

        grouped_data_keys = list(self.grouped_data.keys())
        lengths_groups = [len(v) for v in self.grouped_data.values()]
        logger.info(
            f"Obtained {len(grouped_data_keys)} groups by grouping by {self.grouping_fields_repr} : {grouped_data_keys if len(grouped_data_keys) < 10 else grouped_data_keys[:10] + ['...']}, of sizes {lengths_groups if len(lengths_groups) < 10 else lengths_groups[:10] + ['...']} (average {np.mean(lengths_groups):.2f} ± {np.std(lengths_groups):.2f}, runs/group, min {np.min(lengths_groups)}, max {np.max(lengths_groups)})"
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

        # Define label function
        if len(self.grouping_fields) == 0:
            # No grouping, use the label_individual as the key
            def label_fn(group_key, config):
                try:
                    return eval(
                        self.label_individual, {"config": config, "run_name": group_key}
                    )
                except Exception as e:
                    logger.error(
                        f"Could not evaluate label expression '{self.label_individual}', using run name instead - {e}"
                    )
                    return group_key

        elif self.grouping_fields == [None]:
            # Grouping by None, use the method_aggregate as the label
            label_fn = lambda group_key, config: self.method_aggregate
        else:
            # Grouping with multiple groups, use the group key as the label
            def label_fn(group_key, config):
                if len(group_key) == 1:
                    return group_key[0]
                else:
                    return ", ".join(str(k) for k in group_key)

        n_plot = 0
        for group_key, runs_data in self.grouped_data.items():
            # Get the merged DataFrame for all runs in the group
            list_group_dfs: List[pd.DataFrame] = []
            df_x_values = pd.DataFrame()
            max_t = -np.inf
            for run in runs_data:
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
                        logger.error(
                            f"Could not evaluate metric expression '{self.metric}' for run {run['name']}, skipping - {e}"
                        )
                        continue
                else:
                    logger.error(
                        f"Could not find x-axis column '{self.x_axis}' in run {run['name']}, skipping."
                    )
            if not list_group_dfs:  # Skip group if no valid data
                logger.warning(f"Skipping group {group_key} due to missing data.")
                continue
            merged_df = pd.concat(list_group_dfs, axis=1, join="outer")
            x_values = df_x_values[self.x_axis]

            # Get plot kwargs
            plot_kwargs = self.default_plot_kwargs
            if group_key in self.grouped_plot_kwargs:
                plot_kwargs.update(self.grouped_plot_kwargs[group_key])

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
            delta_low = delta_high = None
            if self.method_error == "std":
                delta_low = delta_high = merged_df.std(axis=1, skipna=True)
            elif self.method_error == "sem":
                delta_low = delta_high = merged_df.sem(axis=1, skipna=True)
            elif self.method_error == "range":
                delta_low = mean_values - merged_df.min(axis=1, skipna=True)
                delta_high = merged_df.max(axis=1, skipna=True) - mean_values
            elif self.method_error == "iqr":
                q1 = merged_df.quantile(0.25, axis=1)
                q3 = merged_df.quantile(0.75, axis=1)
                delta_low = mean_values - q1
                delta_high = q3 - mean_values
            elif self.method_error.startswith("percentile"):
                q = float(self.method_error.split("_")[-1])
                delta_low = mean_values - merged_df.quantile(q / 2, axis=1)
                delta_high = merged_df.quantile(1 - q / 2, axis=1) - mean_values
            elif self.method_error == "none":
                pass
            else:
                raise ValueError(f"Invalid method_error: {self.method_error}")

            # Plot aggregated values
            plt.plot(
                x_values,
                values_aggregated,
                label=(
                    str(label_fn(group_key=group_key, config=runs_data[0]["config"]))
                    if n_plot < self.max_legend_length
                    else None
                ),
                **plot_kwargs,
            )
            n_plot += 1
            if delta_low is not None and delta_high is not None:
                plt.fill_between(
                    x_values,
                    mean_values - delta_low,
                    mean_values + delta_high,
                    alpha=self.alpha_shaded,
                    **plot_kwargs,
                )

            # Plot samples
            n = merged_df.shape[1]
            sampled_indices = np.random.choice(
                np.arange(n), size=min(self.n_samples_shown, n), replace=False
            )
            for i in sampled_indices:
                plt.plot(
                    x_values,
                    merged_df.iloc[:, i],
                    alpha=self.alpha_shaded,
                    **plot_kwargs,
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
            logger.error("No data to plot.")
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
        if len(self.grouping_fields) == 0:
            title_legend = None
        elif n_plot > self.max_legend_length:
            title_legend = (
                f"Groups (first {self.max_legend_length} shown / {n_plot} total)"
            )
        else:
            title_legend = "Groups"
        plt.legend(loc="best", fontsize="small", title=title_legend)
        plt.grid(**self.kwargs_grid)
        if self.method_error == "none":
            string_methods_agg_error = f"{self.method_aggregate}"
        else:
            string_methods_agg_error = f"{self.method_aggregate}, σ={self.method_error}"
        if len(self.grouping_fields) == 0 or self.grouping_fields == [None]:
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
        logger.info(f"Found {len(self.run_dirs)} runs in {self.runs_path}.")

        # Load and filter run data
        self.load_and_filter_run_data()
        if len(self.runs_data) == 0:
            logger.error("No runs to plot.")
            return

        # Group runs based on specified fields
        if len(self.grouping_fields) > 0 or True:
            self.group_runs()
        else:
            self.grouped_data = {run["name"]: [run] for run in self.runs_data}
        if sum(len(v) for v in self.grouped_data.values()) == 0:
            raise ValueError(
                "Run were loaded but no runs were grouped. This should not happen."
            )

        # Plot grouped data
        self.plot_grouped_data()
        logger.info("Plotting complete.")


@hydra.main(
    config_path="../configs_plot",
    config_name="config_default.yaml",
    version_base="1.3.2",
)
def main(config: DictConfig):
    """Main function to initialize and run the Plotter."""
    plotter = Plotter(config)
    plotter.run()


if __name__ == "__main__":
    main()
