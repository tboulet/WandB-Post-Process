import datetime
import logging
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
from typing import List, Dict, Any, Optional, Set, Tuple, Type, Union
import enum

KWARGS_IMPORTS = {"np": np, "pd": pd, "re": re}

# Representation functions
join = lambda *args: ", ".join(args)
join_par = lambda *args: f"({join(*args)})" if len(args) > 1 else join(*args)
drop_prefix = lambda s: re.sub(r"\b\w+\.(\w+)\b", r"\1", s)


class EvalArgs(enum.Enum):
    """Enum class for the arguments that can be passed to the eval function."""

    CONFIG = "config"
    RUN_NAME = "run_name"
    METRIC = "metric"
    X_VALUES = "x_values"
    METRICS = "metrics"
    METRIC_EXPRESSION = "metric_expression"
    GROUP_KEY = "group_key"
    JOIN_FN = "join"
    JOIN_PAR_FN = "join_par"
    DROP_PREFIX_FN = "drop_prefix"


class RunMetricData:
    """This class contains the data of one particular metric for one particular run."""

    def __init__(
        self,
        run_name: str,
        metric_expression: str,
        x_values: np.ndarray,
        metric_values: np.ndarray,
        config: Dict[str, Any],
        group_key: Tuple = (),
    ):
        """Initializes the RunMetricData class.

        Args:
            run_name (str): the name of the run.
            metric_name (str): the name of the metric considered.
            x_values (pd.Series): the x values of the run.
            metric_values (pd.Series): the values of the metric considered.
            config (Dict[str, Any]): the configuration of the run.
        """
        self.run_name = run_name
        self.metric_name = metric_expression
        self.x_values = x_values
        self.metric_values = metric_values
        self.config = config
        self.group_key = group_key


class GroupIndexer:
    """A class that manages the indexes of each group in the scope of the all plot."""

    def __init__(self, groups: List[Dict[str, Any]]):
        self.groups = groups
        self.expression_group_to_index_mapping = {
            dict_group["expression"]: {} for dict_group in groups
        }

    def add_group(self, group_key: Tuple):
        """Add a group to the index mapping."""
        for i, (group_subkey, dict_group) in enumerate(zip(group_key, self.groups)):
            expression_group = dict_group["expression"]
            if (
                group_subkey
                not in self.expression_group_to_index_mapping[expression_group]
            ):
                self.expression_group_to_index_mapping[expression_group][
                    group_subkey
                ] = len(self.expression_group_to_index_mapping[expression_group])

    def get_group_index(self, group_key: Tuple) -> Tuple[int]:
        """Get the index of a group."""
        return tuple(
            self.expression_group_to_index_mapping[dict_group["expression"]][
                group_subkey
            ]
            for group_subkey, dict_group in zip(group_key, self.groups)
        )

    def get_max_indexes(self) -> Tuple[int]:
        """Get the maximum index for each group."""
        return tuple(
            len(self.expression_group_to_index_mapping[dict_group["expression"]])
            for dict_group in self.groups
        )

    def get_group_kwargs_plot(self, group_key) -> Dict:
        """Get the kwargs_plot for a group, from the args_plot specified in the groups if any."""
        kwargs_plot = {}
        for group_subkey, dict_group in zip(group_key, self.groups):
            if "key_plot" in dict_group and "args_plot" in dict_group:
                key_plot = dict_group["key_plot"]
                args_plot = dict_group["args_plot"]
                if len(args_plot) == 0:
                    continue
                idx_subgroup = self.expression_group_to_index_mapping[
                    dict_group["expression"]
                ][group_subkey]
                idx_subgroup_modulo = idx_subgroup % len(args_plot)
                kwargs_plot[key_plot] = args_plot[idx_subgroup_modulo]
        return kwargs_plot

    def get_x_and_width(self, group_key: Tuple, x_min: float, x_max: float) -> Tuple:
        """Get the x values and the width for a group."""
        return self.get_x_and_width_from_index(
            self.get_group_index(group_key), self.get_max_indexes(), x_min, x_max
        )

    def get_x_and_width_from_index(
        self, group_index: Tuple, group_index_max: Tuple, x_min: float, x_max: float
    ) -> Tuple:
        print(group_index, group_index_max, x_min, x_max)
        """Get the x values and the width for a group."""
        assert len(group_index) == len(group_index_max), "Invalid group index."
        if len(group_index) == 0:
            raise ValueError("Empty group index.")
        elif len(group_index) == 1:
            index = group_index[0]
            index_max = group_index_max[0]
            n_indexes = index_max + 1
            percentage_width = 0.8
            width = (x_max - x_min) * percentage_width / n_indexes
            x = x_min + (1 - percentage_width) / 2 + width * index
            return x, width
        else:
            index_considered = group_index[0]
            index_max_considered = group_index_max[0]
            n_indexes_considered = index_max_considered + 1
            width_allocated = (x_max - x_min) / n_indexes_considered
            x_min_next = x_min + width_allocated * index_considered
            x_max_next = x_min + width_allocated * (index_considered + 1)
            return self.get_x_and_width_from_index(
                group_index[1:], group_index_max[1:], x_min_next, x_max_next
            )


class WandbppPlotter:

    def __init__(self, config: DictConfig):
        """Initializes the Plotter class with configuration parameters.

        Args:
            config (DictConfig): Hydra configuration object.
        """
        self.config = dict(config)
        # Set logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="[WandbPP plot] %(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
        # Date
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Paths
        self.runs_path = self.config.get("runs_path", "data/wandb_exports")
        self.file_scalars: str = self.config.get("file_scalars", "scalars.csv")
        self.file_config: str = self.config.get("file_config", "config.yaml")
        # Metric expressions
        self.metrics_expressions: List[str] = self.config.get("metrics_expressions", [])
        assert (
            len(self.metrics_expressions) > 0
        ), "No metric(s) specified. You must specify your metrics in the 'metrics_expressions' field."
        self.metrics_expressions_repr = drop_prefix(join(*self.metrics_expressions))
        # Filter expressions
        self.filters_expressions: List[str] = self.config.get("filters_expressions", [])
        self.filters_args: Set[str] = {
            eval_arg
            for eval_arg in EvalArgs
            if eval_arg.value in " ".join(self.filters_expressions)
        }
        # Grouping expressions
        self.groups: List[Dict[str, Any]] = self.config.get("groups", [])
        for i, dict_group in enumerate(self.groups):
            if isinstance(dict_group, str):
                self.groups[i] = {"expression": dict_group}
        self.groups_expressions: List[str] = [
            group["expression"] for group in self.groups
        ]
        self.groups_args: Set[str] = {
            eval_arg
            for eval_arg in EvalArgs
            if eval_arg.value in " ".join(self.groups_expressions)
        }
        self.groups_repr = drop_prefix(join_par(*self.groups_expressions))
        self.group_indexer = GroupIndexer(self.groups)
        # Aggregation options
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
        self.max_n_curves: int = self.config.get("max_n_curves", np.inf)
        if self.max_n_runs in [None, "None", "null", "inf", "np.inf"]:
            self.max_n_runs = np.inf
        # Plotting interface options
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
        self.kwargs_plot: Dict = self.config.get("kwargs_plot", {})
        self.do_show_one_by_one: bool = self.config.get("do_show_one_by_one", False)
        self.figsize: List[int] = self.config.get("figsize", [10, 6])
        # Show/save options
        self.do_show: bool = self.config.get("do_show", True)
        self.do_save: bool = self.config.get("do_save", False)
        self.path_save: str = self.config.get(
            "path_save", "f'plots/plot_{metric}_{date}.png'"
        )

    def load_grouped_data(self, run_dirs: List[str]) -> Dict[str, List[RunMetricData]]:
        grouped_data: Dict[str, List[RunMetricData]] = {}
        n_run_loaded = 0
        n_curves_showed = 0
        for run_path in tqdm(run_dirs, desc="[WandbPP plot] Filtering ..."):
            if n_run_loaded >= self.max_n_runs:
                break
            # Get run name from run_path
            run_name = os.path.basename(run_path)
            # Load run config
            config_path = os.path.join(run_path, self.file_config)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    run_config = yaml.safe_load(f)
            else:
                run_config = {}
                self.logger.warning(
                    f"[Run Loading] Could not find config file {self.file_config} in {run_path}. Using empty config for this run."
                )
            run_config = OmegaConf.create(run_config)
            # If filters contain only "config" or nothing (case 1), apply filters using only config
            if self.filters_args in [set(), {EvalArgs.CONFIG}]:
                if not self.apply_filters(run_name=run_name, run_config=run_config):
                    continue
            # If grouping contain only "config" or nothing (case A), apply grouping using only config
            if self.groups_args in [set(), {EvalArgs.CONFIG}]:
                group_key = self.get_group_key(run_name=run_name, run_config=run_config)
            # Load scalars file
            scalars_path = os.path.join(run_path, self.file_scalars)
            if not os.path.exists(scalars_path):
                self.logger.error(
                    f"Could not find scalars file {self.file_scalars} in {run_path}. Skipping run."
                )
                return {}
            file_scalars = open(scalars_path, "r")
            metrics = pd.read_csv(file_scalars)
            file_scalars.close()
            # Extract x values
            if not self.x_axis in metrics.columns:
                self.logger.error(
                    f"Could not find x-axis column '{self.x_axis}' in run {run_name}. Skipping run."
                )
                return {}
            x_values = metrics[self.x_axis]
            # If filters don't contain "metric" and "x_values" but contain "metrics" (case 2), apply filters using only config and metrics
            if (
                EvalArgs.METRIC not in self.filters_args
                and EvalArgs.X_VALUES not in self.filters_args
                and EvalArgs.METRICS in self.filters_args
            ):
                if not self.apply_filters(
                    run_name=run_name,
                    run_config=run_config,
                    metrics=metrics,
                ):
                    return {}
            # If grouping don't contain "metric" and "x_values" but contain "metrics" (case B), apply grouping using only config and metrics
            if (
                EvalArgs.METRIC not in self.groups_args
                and EvalArgs.X_VALUES not in self.groups_args
                and EvalArgs.METRICS in self.groups_args
            ):
                group_key = self.get_group_key(
                    run_name=run_name,
                    run_config=run_config,
                    metrics=metrics,
                )
            # Iterate on metrics
            is_curve_added = False
            for metric_expression in self.metrics_expressions:
                # Break if max number of runs reached
                if n_curves_showed >= self.max_n_curves:
                    break
                # Evaluate metric expression
                try:
                    metric_values = eval(
                        metric_expression,
                        {
                            EvalArgs.CONFIG.value: run_config,
                            EvalArgs.X_VALUES.value: x_values,
                            EvalArgs.METRICS.value: metrics,
                            **KWARGS_IMPORTS,
                        },
                    )
                except Exception as e:
                    self.logger.error(
                        f"Could not evaluate metric expression '{metric_expression}' for run {run_name}, skipping this expression - {e}"
                    )
                    continue
                # Assert obtained metric values are of a valid type
                valid_types = [pd.Series, np.float64]
                if not isinstance(metric_values, tuple(valid_types)):
                    self.logger.error(
                        f"Invalid type for metric {metric_expression}: {type(metric_values)} is not in {valid_types}. Skipping this expression."
                    )
                    continue
                # If filters contain "metric" (case 3), apply filters using all arguments
                if EvalArgs.METRIC in self.filters_args:
                    if not self.apply_filters(
                        run_name=run_name,
                        run_config=run_config,
                        metric_expression=metric_expression,
                        metric_values=metric_values,
                        x_values=x_values,
                        metrics=metrics,
                    ):
                        continue
                # If grouping contain "metric" or "metric_expression" (case C), apply grouping using all arguments
                if (
                    EvalArgs.METRIC in self.groups_args
                    or EvalArgs.METRIC_EXPRESSION in self.groups_args
                ):
                    group_key = self.get_group_key(
                        run_name=run_name,
                        run_config=run_config,
                        metric_expression=metric_expression,
                        metric_values=metric_values,
                        x_values=x_values,
                        metrics=metrics,
                    )
                # Create RunMetricData object
                run_metric_data = RunMetricData(
                    run_name=run_name,
                    metric_expression=metric_expression,
                    x_values=x_values,
                    metric_values=metric_values,
                    config=run_config,
                    group_key=group_key,
                )
                if not group_key in grouped_data:
                    grouped_data[group_key] = []
                grouped_data[group_key].append(run_metric_data)
                # Add group to the group indexer
                self.group_indexer.add_group(group_key)
                # Increment number of curves showed and run loaded
                n_curves_showed += 1
                is_curve_added = True
            # Increment number of run loaded
            if is_curve_added:
                n_run_loaded += 1
        lengths_groups = [len(v) for v in grouped_data.values()]
        grouped_data_keys = list(grouped_data.keys())
        if len(grouped_data_keys) > 0:
            self.logger.info(
                f"After filtering and grouping : {n_run_loaded} runs loaded, {n_curves_showed} curves showed."
            )
            self.logger.info(
                f"Obtained {len(grouped_data_keys)} groups by grouping by {self.groups_repr} : {grouped_data_keys if len(grouped_data_keys) < 10 else grouped_data_keys[:10] + ['...']}, of sizes {lengths_groups if len(lengths_groups) < 10 else lengths_groups[:10] + ['...']} (average {np.mean(lengths_groups):.2f} ± {np.std(lengths_groups):.2f}, runs/group, min {np.min(lengths_groups)}, max {np.max(lengths_groups)})"
            )
        else:
            self.logger.error("No runs were loaded.")
        return grouped_data

    def apply_filters(
        self,
        run_name: str,
        run_config: Dict[str, Any],
        metric_expression: Optional[str] = "no metric (no-metric condition)",
        metric_values: Optional[np.ndarray] = None,
        x_values: Optional[np.ndarray] = None,
        metrics: Optional[pd.DataFrame] = None,
    ):
        """Applies filters on a RunMetricData components to determine if it should be included in the data.
        A RunMetricData corresponds to a specific metric for a specific run.
        RunMetricData components involves the config of the run, the "metric" the y_values of the metric on this run, "x" the x_values of the run, and can also involve metrics (the whole run dataframe).
        All these fields except the run_name and config are optional for this method, but they must appear if they appear in the filters.

        Args:
            run_name (str): The name of the run.
            run_config (Dict[str, Any]): The config of the run.
            metric_expression (str): The name of the metric.
            metric (Optional[np.ndarray]): The values of the metric on this run.
            x_values (Optional[np.ndarray]): The x_values of the run.
            metrics (Optional[pd.DataFrame]): The run dataframe.

        Returns:
            bool: True if run satisfies all filters, False otherwise.
        """
        for filter_condition in self.filters_expressions:
            try:
                if not eval(
                    filter_condition,
                    {
                        EvalArgs.CONFIG.value: run_config,
                        EvalArgs.METRIC.value: metric_values,
                        EvalArgs.X_VALUES.value: x_values,
                        EvalArgs.METRICS.value: metrics,
                        EvalArgs.METRIC_EXPRESSION.value: metric_expression,
                        **KWARGS_IMPORTS,
                    },
                ):
                    return False
            except Exception as e:
                self.logger.warning(
                    f"[Filtering] Could not evaluate filter condition '{filter_condition}' for run {run_name} metric {metric_expression}, skipping - {e}"
                )
                return False
        return True

    def get_group_key(
        self,
        run_name: str,
        run_config: Dict[str, Any],
        metric_expression: Optional[str] = "no metric (no-metric condition)",
        metric_values: Optional[np.ndarray] = None,
        x_values: Optional[np.ndarray] = None,
        metrics: Optional[pd.DataFrame] = None,
    ) -> Tuple:
        """Constructs a group key based on a RunMetricData components.
        A RunMetricData corresponds to a specific metric for a specific run.
        RunMetricData components involves the config of the run, the "metric" the y_values of the metric on this run, "x" the x_values of the run, and can also involve metrics (the whole run dataframe).
        All these fields except the run_name and config and run_name are optional for this method, but they must appear if they appear in the grouping.

        Args:
            run_name (str): The name of the run.
            run_config (Dict[str, Any]): The config of the run.
            metric_expression (str): The expression of the metric.
            metric (Optional[np.ndarray]): The values of the metric on this run.
            x_values (Optional[np.ndarray]): The x_values of the run.
            metrics (Optional[pd.DataFrame]): The run dataframe.

        Returns:
            Tuple: The group key.
        """
        group_key = []
        for dict_group in self.groups:
            expression_group = dict_group["expression"]
            try:
                value = eval(
                    expression_group,
                    {
                        EvalArgs.CONFIG.value: run_config,
                        EvalArgs.METRIC.value: metric_values,
                        EvalArgs.X_VALUES.value: x_values,
                        EvalArgs.METRICS.value: metrics,
                        EvalArgs.METRIC_EXPRESSION.value: metric_expression,
                    },
                )

            except Exception as e:
                self.logger.warning(
                    f"[Grouping] Could not evaluate grouping expression '{expression_group}' for run {run_name} and metric expression {metric_expression}, skipping curve - {e}"
                )
                value = "None"  # assign 'None' if field evaluation fails
            group_key.append(value)
        group_key = tuple(group_key)  # use tuple for hashability
        return group_key

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

    def treat_grouped_data(self, grouped_data: Dict[Any, List[RunMetricData]]):
        """Plots metrics with mean and standard error for each group.
        Also defines y_min and y_max based on the metric values.
        """

        # Define label function
        def label_fn(group_key: Tuple):
            group_key_pretty_list = []
            for elem in group_key:
                if isinstance(elem, str):
                    elem = drop_prefix(elem)
                else:
                    elem = str(elem)
                group_key_pretty_list.append(elem)
            group_key_pretty = join(*group_key_pretty_list)
            return group_key_pretty

        # Iterate on the groups
        plt.figure(figsize=self.figsize)
        y_min, y_max = np.inf, -np.inf
        x_max = max(
            max(
                run_metric_data.x_values.max()
                for run_metric_data in list_runs_metric_data
            )
            for list_runs_metric_data in grouped_data.values()
        )
        x_min = min(
            min(
                run_metric_data.x_values.min()
                for run_metric_data in list_runs_metric_data
            )
            for list_runs_metric_data in grouped_data.values()
        )
        for group_key, list_runs_metric_data in grouped_data.items():
            # Check if non empty group
            if not list_runs_metric_data:
                self.logger.warning(
                    f"[Aggregating] Skipping group {group_key} due to missing data."
                )
                continue
            n_curve_plotted = 0
            # Get the merged DataFrame for all runs in the group
            list_metric_values: List[pd.Series] = []
            x_values_global: pd.Series = pd.Series()
            max_t = -np.inf
            for run_metric_data in list_runs_metric_data:
                # Get the metric values
                metric_values = run_metric_data.metric_values
                list_metric_values.append(metric_values)
                # Get the x-axis values global
                x_values = run_metric_data.x_values
                if max_t < x_values.max():
                    max_t = x_values.max()
                    x_values_global = x_values
            if len(list_metric_values) == 0:  # Skip group if no valid data
                self.logger.warning(
                    f"[Aggregating] Skipping group {group_key} due to missing data."
                )
                continue
            # Check type of each element of list_metric_values and apply adapted operation
            type_metric_values = self.get_general_type(list_metric_values[0])
            if not all(
                self.get_general_type(elem) == type_metric_values
                for elem in list_metric_values
            ):
                self.logger.error(
                    f"[Aggregating] Inconsistent types in list_metric_values: {[type(elem) for elem in list_metric_values]}. Skipping group {group_key}."
                )
                continue
            if type_metric_values == pd.Series:
                merged_df = pd.concat(list_metric_values, axis=1, join="outer")
            elif type_metric_values == np.float64:
                merged_df = pd.DataFrame(list_metric_values).T
            else:
                self.logger.error(
                    f"[Aggregating] Invalid type for metric values: {type_metric_values}. Skipping group {group_key}."
                )
                continue

            # Get plot kwargs
            kwargs_plot_group = self.kwargs_plot.copy()
            for group_subkey, dict_group in zip(group_key, self.groups):
                if "key_plot" in dict_group:
                    if "kwargs_plot" not in dict_group:
                        self.logger.warning(
                            f"Group {expression_group} has a key_plot but no kwargs_plot. Skipping the label."
                        )
                        continue
                else:
                    continue
                key_plot = dict_group["key_plot"]
                expression_group = dict_group["expression"]
                kwargs_plot = dict_group["kwargs_plot"]
                group_subkey_repr = str(group_subkey)
                if group_subkey_repr not in kwargs_plot:
                    self.logger.warning(
                        f"Category {group_subkey_repr} not in kwargs_plot for group {expression_group}. Skipping the label."
                    )
                    continue
                kwargs_plot_group[key_plot] = kwargs_plot[group_subkey_repr]

            # Get group label
            label = (
                label_fn(group_key)
                if n_curve_plotted < self.max_legend_length
                else None
            )

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

            # Plot aggregated values (curves)
            if type_metric_values == pd.Series:
                plt.plot(
                    x_values_global,
                    values_aggregated,
                    label=label,
                    **kwargs_plot_group,
                )
                n_curve_plotted += 1
                if delta_low is not None and delta_high is not None:
                    plt.fill_between(
                        x_values_global,
                        mean_values - delta_low,
                        mean_values + delta_high,
                        alpha=self.alpha_shaded,
                        **kwargs_plot_group,
                    )

                # Plot samples
                n = merged_df.shape[1]
                sampled_indices = np.random.choice(
                    np.arange(n), size=min(self.n_samples_shown, n), replace=False
                )
                for i in sampled_indices:
                    plt.plot(
                        x_values_global,
                        merged_df.iloc[:, i],
                        alpha=self.alpha_shaded,
                        **kwargs_plot_group,
                    )
                    
            elif type_metric_values == np.float64:
                # Plot the bars for aggregated values
                x, width = self.group_indexer.get_x_and_width(
                    group_key, x_min, x_max
                )
                plt.bar(
                    [x],
                    values_aggregated[0],
                    yerr=[[delta_low[0]], [delta_high[0]]],
                    label=label,
                    width=width,
                    **kwargs_plot_group,
                )
                n = merged_df.shape[1]
                sampled_indices = np.random.choice(
                    np.arange(n), size=min(self.n_samples_shown, n), replace=False
                )
                for i in sampled_indices:
                    plt.hlines(
                        merged_df.iloc[:, i],
                        x - width / 2 * 1.2,
                        x + width / 2 * 1.2,
                        alpha=self.alpha_shaded,
                        **kwargs_plot_group,
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
            self.logger.error("[Plotting] No data to plot.")
            return

        self.logger.info(
            f"[Plotting] Building plot for {len(grouped_data)} aggregated curves."
        )

        # ======= Plot settings =======
        plt.xlabel(self.x_axis)
        plt.ylabel(self.metrics_expressions_repr)
        # Set x and y limits
        plt.xlim(self.x_lim)
        if self.do_try_include_y0:
            y_lim = self.try_include_y0(self.y_lim, y_min=y_min, y_max=y_max)
        else:
            y_lim = self.y_lim
        plt.ylim(y_lim)
        # Legend and grid
        if len(self.groups) == 0:
            title_legend = None
        elif n_curve_plotted > self.max_legend_length:
            title_legend = f"Groups (first {self.max_legend_length} shown / {n_curve_plotted} total)"
        else:
            title_legend = "Groups"
        plt.legend(title=title_legend, **self.config.get("kwargs_legend", {}))
        plt.grid(**self.config.get("kwargs_grid", {}))
        # Title
        if self.method_error == "none":
            string_methods_agg_error = f"{self.method_aggregate}"
        else:
            string_methods_agg_error = f"{self.method_aggregate}, σ={self.method_error}"
        if len(self.groups) == 0:
            title = f"{self.metrics_expressions_repr} ({string_methods_agg_error})"
        else:
            title = f"{self.metrics_expressions_repr} (grouped by {self.groups_repr} : {string_methods_agg_error})"
        plt.title(title, **self.config.get("kwargs_title", {}))

        # Show plot
        if self.do_show:
            plt.show()

        # Save plot
        if self.do_save:
            metric_repr_file_compatible = self.metrics_expressions_repr.replace(
                "/", "_per_"
            )
            try:
                path_save = eval(
                    self.path_save,
                    {"metric": metric_repr_file_compatible, "date": self.date},
                )
            except Exception as e:
                path_save = f"plots/plot_{metric_repr_file_compatible}_{self.date}.png"
                self.logger.error(
                    f"Could not evaluate path_save expression '{self.path_save}', using default path '{path_save}' instead - {e}"
                )
            path_save = self.sanitize_filepath(path_save)
            os.makedirs(os.path.dirname(path_save), exist_ok=True)
            plt.savefig(path_save, **self.config.get("kwargs_savefig", {}))
            self.logger.info(
                f"""Saved plot for metrics "{self.metrics_expressions_repr}" to {path_save}"""
            )

    # ======= Utility functions =======
    def get_general_type(self, value: Any) -> Type:
        """Returns the general type of a value."""
        if np.isscalar(value):
            return np.float64
        elif isinstance(value, pd.Series):
            return pd.Series
        else:
            return type(value)

    # ======= Main function =======
    def run(self):
        """Executes the full pipeline: loading, filtering, grouping, and plotting."""
        run_dirs = [
            os.path.join(self.runs_path, d)
            for d in os.listdir(self.runs_path)
            if os.path.isdir(os.path.join(self.runs_path, d))
        ]
        self.logger.info(f"Found {len(run_dirs)} runs in {self.runs_path}.")

        # Load and filter run data
        grouped_data = self.load_grouped_data(run_dirs=run_dirs)

        # Plot grouped data
        self.treat_grouped_data(grouped_data=grouped_data)
        self.logger.info("End.")


@hydra.main(
    config_path="../configs_plot",
    config_name="config_default.yaml",
    version_base="1.3.2",
)
def main(config: DictConfig):
    """Main function to initialize and run the Plotter."""
    plotter = WandbppPlotter(config)
    plotter.run()


if __name__ == "__main__":
    main()
