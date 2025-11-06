import os
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

from data_manager import DataManager

class Visualizer:
    def __init__(self, config, data_manager: DataManager):
        self.config = config.get_config()
        self.dm = data_manager
        self.exp = self.dm.experiment
        self.output_dir = self.config['img']['plot_output_dir']
        
        # --- Professional Plotting Style ---
        plt.style.use('seaborn-v0_8-paper')
        self.colors = plt.get_cmap('tab10').colors
        plot_config = self.config.get('plot', {})
        try:
            stride_value = int(plot_config.get('frame_stride', 50))
        except (TypeError, ValueError):
            stride_value = 50
        self.frame_stride = max(1, stride_value)

        default_marker_size = 96.0
        marker_size_value = plot_config.get('marker_size', default_marker_size)
        try:
            marker_size = float(marker_size_value)
        except (TypeError, ValueError):
            marker_size = default_marker_size

        marker_alpha_value = plot_config.get('marker_alpha', 0.9)
        try:
            marker_alpha = float(marker_alpha_value)
        except (TypeError, ValueError):
            marker_alpha = 0.9
        marker_alpha = min(max(marker_alpha, 0.0), 1.0)

        marker_edge = plot_config.get('marker_edge', 'black')

        marker_edge_width_value = plot_config.get('marker_edge_width', 0.6)
        try:
            marker_edge_width = float(marker_edge_width_value)
        except (TypeError, ValueError):
            marker_edge_width = 0.6

        line_width_value = plot_config.get('line_width', 1.5)
        try:
            line_width = float(line_width_value)
        except (TypeError, ValueError):
            line_width = 1.5

        line_alpha_value = plot_config.get('line_alpha', 0.7)
        try:
            line_alpha = float(line_alpha_value)
        except (TypeError, ValueError):
            line_alpha = 0.7
        line_alpha = min(max(line_alpha, 0.0), 1.0)

        self.scatter_params = {
            's': marker_size,
            'alpha': marker_alpha,
            'marker': plot_config.get('marker', 'o'),
            'edgecolors': marker_edge,
            'linewidths': marker_edge_width,
        }
        self.line_params = {
            'linewidth': line_width,
            'alpha': line_alpha,
        }
        self.time_axis_label = plot_config.get('time_label', 'Time (ms)')
        self.font_sizes = {
            'title': 16,
            'label': 12,
            'legend': 10,
        }

    def _setup_plot(self, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title(title, fontsize=self.font_sizes['title'], fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.font_sizes['label'])
        ax.set_ylabel(ylabel, fontsize=self.font_sizes['label'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        return fig, ax

    def _save_plot(self, fig, exp_id, filename):
        if exp_id is not None:
            exp_name = self.exp.experiment_names[exp_id]
            plot_dir = os.path.join(self.output_dir, exp_name)
        else:
            plot_dir = os.path.join(self.output_dir, "comparison")

        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved plot to {save_path}")

    def _to_numpy_sequence(self, data):
        if isinstance(data, range):
            return np.arange(data.start, data.stop, data.step, dtype=int)
        return np.asarray(data)

    def _stride_indices(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=int)
        stride = max(1, self.frame_stride)
        indices = np.arange(0, length, stride, dtype=int)
        if indices[-1] != length - 1:
            indices = np.append(indices, length - 1)
        return indices

    def _time_series_ms(self, exp_id: int) -> np.ndarray:
        time_values = self.exp.result_data['time'][exp_id]
        if not time_values:
            return np.array([], dtype=float)
        return np.asarray(time_values, dtype=float) * 1e3

    def _scatter_series(self, ax, frames, values, *, color, label):
        frames_arr = self._to_numpy_sequence(frames)
        values_arr = self._to_numpy_sequence(values)
        if frames_arr.size == 0 or values_arr.size == 0:
            return
        indices = self._stride_indices(len(frames_arr))
        frames_ds = frames_arr[indices]
        values_ds = values_arr[indices]

        ax.plot(frames_ds, values_ds, color=color, **self.line_params)
        ax.scatter(frames_ds, values_ds, color=color, label=label, **self.scatter_params)

    def _scatter_multi_series(self, ax, frames, values, labels, color_offset: int = 0):
        frames_arr = self._to_numpy_sequence(frames)
        values_arr = np.asarray(values)
        if frames_arr.size == 0 or values_arr.size == 0:
            return
        if values_arr.ndim == 1:
            values_arr = values_arr[:, np.newaxis]
        indices = self._stride_indices(frames_arr.shape[0])
        frames_ds = frames_arr[indices]
        for i, label in enumerate(labels):
            if i >= values_arr.shape[1]:
                break
            series = values_arr[:, i]
            if series.size == 0:
                continue
            series_ds = series[indices]
            color = self.colors[(color_offset + i) % len(self.colors)]
            ax.plot(frames_ds, series_ds, color=color, **self.line_params)
            ax.scatter(frames_ds, series_ds, color=color, label=label, **self.scatter_params)

    def plot_area_vs_time(self, exp_id: int):
        area_data = self.exp.result_data['area'][exp_id]
        time_ms = self._time_series_ms(exp_id)
        exp_name = self.exp.experiment_names[exp_id]
        
        fig, ax = self._setup_plot(f"{exp_name}: Contour Area vs. Time", self.time_axis_label, "Contour Area (m^2)")
        self._scatter_series(ax, time_ms, area_data, color=self.colors[0], label='Contour Area')
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "area_vs_time.png")

    def plot_arc_length_vs_time(self, exp_id: int):
        arc_length_data = self.exp.result_data['arc_length'][exp_id]
        time_ms = self._time_series_ms(exp_id)
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Arc Length vs. Time", self.time_axis_label, "Arc Length (m)")
        self._scatter_series(ax, time_ms, arc_length_data, color=self.colors[1], label='Arc Length')
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "arc_length_vs_time.png")

    def plot_area_vec_vs_time(self, exp_id: int):
        area_vec_data = self.exp.result_data['area_vec'][exp_id][1:]
        time_ms = self._time_series_ms(exp_id)
        time_ms = time_ms[1:] if time_ms.size else time_ms
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Area Change vs. Time", self.time_axis_label, "Area Change (m^2/s)")
        self._scatter_series(ax, time_ms, area_vec_data, color=self.colors[2], label='Area Change')
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "area_vec_vs_time.png")

    def plot_regression_circle_radius_vs_time(self, exp_id: int):
        radius_data = self.exp.result_data['regression_circle_radius'][exp_id]
        time_ms = self._time_series_ms(exp_id)
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Regression Circle Radius vs. Time", self.time_axis_label, "Radius (m)")
        self._scatter_series(ax, time_ms, radius_data, color=self.colors[3], label='Radius')
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "regression_circle_radius_vs_time.png")

    def plot_expand_vec_vs_time(self, exp_id: int):
        expand_vel_data = self.exp.result_data['expand_vec'][exp_id]
        sanitized_expand_vel = [item if item else [0, 0, 0, 0] for item in expand_vel_data]
        expand_vel_data_np = np.array(sanitized_expand_vel)
        time_ms = self._time_series_ms(exp_id)

        labels = ['Up', 'Right', 'Down', 'Left']
        exp_name = self.exp.experiment_names[exp_id]
        
        fig, ax = self._setup_plot(f"{exp_name}: Expand Velocity vs. Time", self.time_axis_label, "Velocity (m/s)")
        self._scatter_multi_series(ax, time_ms, expand_vel_data_np, labels, color_offset=4)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "expand_vel_vs_time.png")
    
    def plot_expand_dist_vs_time(self, exp_id: int):
        expand_dist_data = self.exp.result_data['expand_dist'][exp_id]
        sanitized_expand_dist = [item if item else [0, 0, 0, 0] for item in expand_dist_data]
        expand_dist_data_np = np.array(sanitized_expand_dist)
        time_ms = self._time_series_ms(exp_id)

        labels = ['Up', 'Right', 'Down', 'Left']
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Expand Distance vs. Time", self.time_axis_label, "Distance (m)")
        self._scatter_multi_series(ax, time_ms, expand_dist_data_np, labels, color_offset=4)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "expand_dist_vs_time.png")

    def plot_tip_distance_vs_time(self, exp_id: int):
        tip_distance_data = self.exp.result_data['tip_distance'][exp_id]
        time_ms = self._time_series_ms(exp_id)
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Tip Distance vs. Time", self.time_axis_label, "Tip Distance (m)")
        self._scatter_series(ax, time_ms, tip_distance_data, color=self.colors[8], label='Tip Distance')
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "tip_distance_vs_time.png")

    def plot_summary_for_experiment(self, exp_id: int):
        logger.info(f"Generating summary plot for experiment {exp_id}...")
        exp_name = self.exp.experiment_names[exp_id]
        
        data_keys = ['area', 'arc_length', 'area_vec', 'regression_circle_radius', 'expand_dist', 'expand_vec', 'tip_distance']
        data = {key: self.exp.result_data[key][exp_id] for key in data_keys}
        time_ms = self._time_series_ms(exp_id)
        time_ms_vec = time_ms[1:] if time_ms.size else time_ms

        fig, axs = plt.subplots(7, 1, figsize=(12, 35))
        fig.suptitle(f"{exp_name}: Summary Plots", fontsize=self.font_sizes['title'] + 4, fontweight='bold', y=0.99)

        plot_configs = [
            {'ax_idx': 0, 'data_key': 'area', 'time': time_ms, 'ylabel': "Contour Area (m^2)", 'title': "Contour Area", 'color_idx': 0},
            {'ax_idx': 1, 'data_key': 'arc_length', 'time': time_ms, 'ylabel': "Arc Length (m)", 'title': "Arc Length", 'color_idx': 1},
            {'ax_idx': 2, 'data_key': 'area_vec', 'time': time_ms_vec, 'data': data['area_vec'][1:], 'ylabel': "Area Change (m^2/s)", 'title': "Area Change", 'color_idx': 2},
            {'ax_idx': 3, 'data_key': 'regression_circle_radius', 'time': time_ms, 'ylabel': "Radius (m)", 'title': "Regression Circle Radius", 'color_idx': 3},
            {'ax_idx': 6, 'data_key': 'tip_distance', 'time': time_ms, 'ylabel': "Tip Distance (m)", 'title': "Tip Distance", 'color_idx': 8},
        ]

        for conf in plot_configs:
            ax = axs[conf['ax_idx']]
            plot_data = conf.get('data', data[conf['data_key']])
            self._scatter_series(
                ax,
                conf['time'],
                plot_data,
                color=self.colors[conf['color_idx']],
                label=conf['title'],
            )
            ax.set_title(conf['title'], fontsize=self.font_sizes['title'])
            ax.set_ylabel(conf['ylabel'], fontsize=self.font_sizes['label'])
            ax.set_xlabel(self.time_axis_label, fontsize=self.font_sizes['label'])
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(fontsize=self.font_sizes['legend'])

        # Multi-line plots
        multi_line_configs = [
            {'ax_idx': 4, 'data_key': 'expand_dist', 'ylabel': "Distance (m)", 'title': "Expand Distance"},
            {'ax_idx': 5, 'data_key': 'expand_vec', 'ylabel': "Velocity (m/s)", 'title': "Expand Velocity"},
        ]
        labels = ['Up', 'Right', 'Down', 'Left']
        for conf in multi_line_configs:
            ax = axs[conf['ax_idx']]
            raw_data = data[conf['data_key']]
            sanitized_data = np.array([item if item else [0,0,0,0] for item in raw_data])
            self._scatter_multi_series(ax, time_ms, sanitized_data, labels, color_offset=4)
            ax.set_title(conf['title'], fontsize=self.font_sizes['title'])
            ax.set_ylabel(conf['ylabel'], fontsize=self.font_sizes['label'])
            ax.set_xlabel(self.time_axis_label, fontsize=self.font_sizes['label'])
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(fontsize=self.font_sizes['legend'])

        self._save_plot(fig, exp_id, "summary_plot.png")
        logger.success(f"Finished generating summary plot for experiment {exp_id}")

    def plot_all_for_experiment(self, exp_id: int, plot_summary: bool = True, plot_single: bool = True):
        logger.info(f"Generating all plots for experiment {exp_id}...")
        if plot_single:
            self.plot_area_vs_time(exp_id)
            self.plot_arc_length_vs_time(exp_id)
            self.plot_area_vec_vs_time(exp_id)
            self.plot_regression_circle_radius_vs_time(exp_id)
            self.plot_expand_dist_vs_time(exp_id)
            self.plot_expand_vec_vs_time(exp_id)
            self.plot_tip_distance_vs_time(exp_id)
        if plot_summary:
            self.plot_summary_for_experiment(exp_id)
        logger.success(f"Finished generating plots for experiment {exp_id}")

    def plot_all(self, plot_summary: bool = True, plot_single: bool = True):
        logger.info("Generating plots for all experiments...")
        num_experiments = len(self.exp.experiment_imgs)
        for exp_id in range(num_experiments):
            self.plot_all_for_experiment(exp_id, plot_summary, plot_single)
        
        # After individual plots, generate comparison plots
        if num_experiments > 1:
            comparison_ids = range(num_experiments)
            scalar_keys = ['area', 'arc_length', 'tip_distance', 'regression_circle_radius']
            self.plot_comparison(comparison_ids, scalar_keys)

        logger.success("Finished generating all plots.")

    def plot_comparison(self, exp_ids: list, data_keys: list):
        logger.info(f"Generating comparison plots for experiments: {exp_ids}")

        for key in data_keys:
            title = f"Comparison: {key.replace('_', ' ').title()} vs. Time"
            ylabel = f"{key.replace('_', ' ').title()} (m)" # Basic unit, can be improved
            fig, ax = self._setup_plot(title, self.time_axis_label, ylabel)

            for i, exp_id in enumerate(exp_ids):
                if key in self.exp.result_data and exp_id < len(self.exp.result_data[key]):
                    data = self.exp.result_data[key][exp_id]
                    time_ms = self._time_series_ms(exp_id)
                    if key == 'area_vec':
                        data = data[1:]
                        time_ms = time_ms[1:] if time_ms.size else time_ms
                    exp_name = self.exp.experiment_names[exp_id]
                    self._scatter_series(
                        ax,
                        time_ms,
                        data,
                        color=self.colors[i % len(self.colors)],
                        label=exp_name,
                    )
                else:
                    logger.warning(f"Data for key '{key}' and experiment ID {exp_id} not found. Skipping.")

            ax.legend(fontsize=self.font_sizes['legend'])
            self._save_plot(fig, None, f"comparison_{key}.png")
