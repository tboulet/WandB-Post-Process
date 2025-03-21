import datetime
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
        # Set logger
        basicConfig(
            level=logging.INFO,
            format="[WandbPP plot] %(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
        # Date
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Data selection options
        self.config = dict(config)
        self.runs_path = self.config.get("runs_path", "data/wandb_exports")
        self.file_scalars: str = self.config.get("file_scalars", "scalars.csv")
        self.file_config: str = self.config.get("file_config", "config.yaml")
        metric: str = self.config.get("metric", None)
        list_metrics: List[str] = self.config.get("metrics", None)
        self.metrics: List[str] = []
        if metric is not None:
            self.metrics.append(metric)
        if list_metrics is not None:
            self.metrics.extend(list_metrics)
        assert (
            len(self.metrics) > 0
        ), "No metric(s) specified. You must specify your metrics either in the config in 'metric' or 'metrics' (list)"
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
        self.label_group: str = self.config.get("label_group", None)
        self.title: str = self.config.get("title", None)
        self.x_axis: str = self.config.get("x_axis", "_step")
        self.x_lim = self.config.get("x_lim", None)
        if self.x_lim is None:
            self.x_lim = [None, None]
        self.y_lim = self.config.get("y_lim", None)
        if self.y_lim is None:
            self.y_lim = [None, None]
        self.do_try_include_y0: bool = self.config.get("do_try_include_y0", False)
        self.ratio_near_y0: str = self.config.get("ratio_near_y0", 0.5)
        self.alpha_shaded: float = self.config.get("alpha_shaded", 0.2)
        self.max_legend_length: int = self.config.get("max_legend_length", 10)
        self.groups_plot_kwargs: List[Dict[str, Any]] = self.config.get(
            "groups_plot_kwargs", {}
        )
        self.default_plot_kwargs: Dict = self.config.get("default_plot_kwargs", {})
        self.do_show_one_by_one: bool = self.config.get("do_show_one_by_one", False)
        self.figsize: List[int] = self.config.get("figsize", [10, 6])
        # Show/save options
        self.do_show: bool = self.config.get("do_show", True)
        self.do_save: bool = self.config.get("do_save", False)
        self.path_save: str = self.config.get(
            "path_save", "f'plots/plot_{metric}_{date}.png'"
        )
        # Define variables
        self.run_dirs: List[str] = []
        self.runs_data: List[Dict[str, Any]] = []
        self.grouped_data: Dict[Any, List[Dict[str, Any]]] = {}
        self.grouped_plot_kwargs: Dict[Any, Dict[str, Any]] = {}

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
                logger.warning(
                    f"Warning: Could not evaluate filter field '{filter_condition}' - {e}"
                )
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
                        f"Could not evaluate grouping field '{field}' for run {run_name}, skipping - {e}"
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

    def sanitize_filepath(self, filepath: str, replacement="_"):
        """
        Sanitize a file path by replacing invalid characters in the filename
        while optionally replacing directory separators.

        Args:
            filepath (str): The original file path.
            replacement (str, optional): Character to replace invalid ones. Defaults to "_".

        Returns:
            str: A sanitized file path safe for saving.
        """
        # Define invalid characters for filenames (excluding / and \ for now)
        invalid_chars = r'[<>:"|?*]'

        # Separate the directory path and the filename
        directory, filename = os.path.split(filepath)

        # Sanitize the filename
        sanitized_filename = re.sub(invalid_chars, replacement, filename)
        sanitized_filename = re.sub(
            rf"{re.escape(replacement)}+", replacement, sanitized_filename
        )
        sanitized_filename = sanitized_filename.strip(replacement)

        # Reconstruct the full path
        sanitized_path = os.path.join(directory, sanitized_filename)

        return sanitized_path

    def plot_grouped_data(self):
        """Plots metrics with mean and standard error for each group.
        Also defines y_min and y_max based on the metric values.
        """

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
                        f"Could not evaluate label_individual expression '{self.label_individual}', using run name instead - {e}"
                    )
                    return group_key

        elif self.grouping_fields == [None]:
            # Grouping by None, use the method_aggregate as the label
            label_fn = lambda group_key, config: self.method_aggregate
        elif self.label_group not in [None, "group_key"]:
            # Grouping with multiple groups, use the label_group as the label
            def label_fn(group_key, config):
                try:
                    return eval(
                        self.label_group, {"config": config, "group_key": group_key}
                    )
                except Exception as e:
                    logger.error(
                        f"Could not evaluate label_group expression '{self.label_group}', using group key instead - {e}"
                    )
                    return group_key

        else:
            # Grouping with multiple groups, use the group key as the label
            def label_fn(group_key, config):
                if len(group_key) == 1:
                    return group_key[0]
                else:
                    return ", ".join(str(k) for k in group_key)

        # Iterate on the metrics
        for metric in self.metrics:
            plt.figure(figsize=self.figsize)
            y_min, y_max = np.inf, -np.inf
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
                            metric_values = eval(metric, {"metrics": df, "np": np})
                            df[metric] = metric_values
                            list_group_dfs.append(df[[metric]])
                            # Get the x-axis values (take last DataFrame as reference)
                            if max_t < df[self.x_axis].max():
                                max_t = df[self.x_axis].max()
                                df_x_values = df[[self.x_axis]]
                        except Exception as e:
                            logger.error(
                                f"Could not evaluate metric expression '{metric}' for run {run['name']}, skipping - {e}"
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
                    raise ValueError(
                        f"Invalid method_aggregate: {self.method_aggregate}"
                    )

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
                        str(
                            label_fn(group_key=group_key, config=runs_data[0]["config"])
                        )
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
                    y_min = min(values_aggregated.min() - delta_low.min(), y_min)
                    y_max = max(values_aggregated.max() + delta_high.max(), y_max)
                else:
                    y_min = min(values_aggregated.min(), y_min)
                    y_max = max(values_aggregated.max(), y_max)

            # Don't plot if no data
            if y_min == np.inf and y_max == -np.inf:
                logger.error("No data to plot.")
                return

            # ======= Plot settings =======
            plt.xlabel("Steps")
            metric_repr = metric.replace("metrics.", "")
            plt.ylabel(metric_repr)
            # Set x and y limits
            plt.xlim(self.x_lim)
            if self.do_try_include_y0:
                y_lim = self.try_include_y0(self.y_lim, y_min=y_min, y_max=y_max)
            plt.ylim(y_lim)
            # Legend and grid
            if len(self.grouping_fields) == 0:
                title_legend = None
            elif n_plot > self.max_legend_length:
                title_legend = (
                    f"Groups (first {self.max_legend_length} shown / {n_plot} total)"
                )
            else:
                title_legend = "Groups"
            plt.legend(title=title_legend, **self.config.get("kwargs_legend", {}))
            plt.grid(**self.config.get("kwargs_grid", {}))
            # Title
            if self.method_error == "none":
                string_methods_agg_error = f"{self.method_aggregate}"
            else:
                string_methods_agg_error = (
                    f"{self.method_aggregate}, σ={self.method_error}"
                )
            if self.title is not None:
                try:
                    title = eval(
                        self.title,
                        {
                            "metric": metric_repr,
                            "grouping_fields_repr": self.grouping_fields_repr,
                        },
                    )
                except Exception as e:
                    if len(self.grouping_fields) == 0 or self.grouping_fields == [None]:
                        title = f"{metric_repr} ({string_methods_agg_error})"
                    else:
                        title = f"{metric_repr} (grouped by {self.grouping_fields_repr} : {string_methods_agg_error})"
                    logger.error(
                        f"Could not evaluate title expression '{self.title}', using default title '{title}' instead - {e}"
                    )
            elif len(self.grouping_fields) == 0 or self.grouping_fields == [None]:
                title = f"{metric_repr} ({string_methods_agg_error})"
            else:
                title = f"{metric_repr} (grouped by {self.grouping_fields_repr} : {string_methods_agg_error})"
            plt.title(title, **self.config.get("kwargs_title", {}))

            # Show plot
            if self.do_show and self.do_show_one_by_one:
                plt.show()

            # Save plot
            if self.do_save:
                metric_repr_file_compatible = metric_repr.replace("/", "_per_")
                try:
                    path_save = eval(
                        self.path_save,
                        {"metric": metric_repr_file_compatible, "date": self.date},
                    )
                except Exception as e:
                    path_save = (
                        f"plots/plot_{metric_repr_file_compatible}_{self.date}.png"
                    )
                    logger.error(
                        f"Could not evaluate path_save expression '{self.path_save}', using default path '{path_save}' instead - {e}"
                    )
                path_save = self.sanitize_filepath(path_save)
                os.makedirs(os.path.dirname(path_save), exist_ok=True)
                plt.savefig(path_save, **self.config.get("kwargs_savefig", {}))
                logger.info(f"Saved metric plot {metric_repr} plot to {path_save}")

        # Show all plots
        if self.do_show and not self.do_show_one_by_one:
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
        if len(self.grouping_fields) > 0:
            self.group_runs()
        else:
            self.grouped_data = {run["name"]: [run] for run in self.runs_data}
        if sum(len(v) for v in self.grouped_data.values()) == 0:
            raise ValueError(
                "Run were loaded but no runs were grouped. This should not happen."
            )

        # Plot grouped data
        self.plot_grouped_data()
        logger.info("End.")


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
