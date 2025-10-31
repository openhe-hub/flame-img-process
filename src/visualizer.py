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
        self.plot_params = {
            'linewidth': 1.5,
            'markersize': 3,
            'marker': 'o',
        }
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

    def plot_area_vs_time(self, exp_id: int):
        area_data = self.exp.result_data['area'][exp_id]
        frames = range(len(area_data))
        exp_name = self.exp.experiment_names[exp_id]
        
        fig, ax = self._setup_plot(f"{exp_name}: Contour Area vs. Time", "Frame Index", "Contour Area (m^2)")
        ax.plot(frames, area_data, color=self.colors[0], label='Contour Area', **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "area_vs_time.png")

    def plot_arc_length_vs_time(self, exp_id: int):
        arc_length_data = self.exp.result_data['arc_length'][exp_id]
        frames = range(len(arc_length_data))
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Arc Length vs. Time", "Frame Index", "Arc Length (m)")
        ax.plot(frames, arc_length_data, color=self.colors[1], label='Arc Length', **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "arc_length_vs_time.png")

    def plot_area_vec_vs_time(self, exp_id: int):
        area_vec_data = self.exp.result_data['area_vec'][exp_id][1:]
        frames = range(1, len(area_vec_data) + 1)
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Area Change vs. Time", "Frame Index", "Area Change (m^2/s)")
        ax.plot(frames, area_vec_data, color=self.colors[2], label='Area Change', **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "area_vec_vs_time.png")

    def plot_regression_circle_radius_vs_time(self, exp_id: int):
        radius_data = self.exp.result_data['regression_circle_radius'][exp_id]
        frames = range(len(radius_data))
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Regression Circle Radius vs. Time", "Frame Index", "Radius (m)")
        ax.plot(frames, radius_data, color=self.colors[3], label='Radius', **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "regression_circle_radius_vs_time.png")

    def plot_expand_vec_vs_time(self, exp_id: int):
        expand_vel_data = self.exp.result_data['expand_vec'][exp_id]
        frames = range(len(expand_vel_data))
        sanitized_expand_vel = [item if item else [0, 0, 0, 0] for item in expand_vel_data]
        expand_vel_data_np = np.array(sanitized_expand_vel)

        labels = ['Up', 'Right', 'Down', 'Left']
        exp_name = self.exp.experiment_names[exp_id]
        
        fig, ax = self._setup_plot(f"{exp_name}: Expand Velocity vs. Time", "Frame Index", "Velocity (m/s)")
        for i in range(expand_vel_data_np.shape[1]):
            ax.plot(frames, expand_vel_data_np[:, i], color=self.colors[i+4], label=labels[i], **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "expand_vel_vs_time.png")
    
    def plot_expand_dist_vs_time(self, exp_id: int):
        expand_dist_data = self.exp.result_data['expand_dist'][exp_id]
        frames = range(len(expand_dist_data))
        sanitized_expand_dist = [item if item else [0, 0, 0, 0] for item in expand_dist_data]
        expand_dist_data_np = np.array(sanitized_expand_dist)

        labels = ['Up', 'Right', 'Down', 'Left']
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Expand Distance vs. Time", "Frame Index", "Distance (m)")
        for i in range(expand_dist_data_np.shape[1]):
            ax.plot(frames, expand_dist_data_np[:, i], color=self.colors[i+4], label=labels[i], **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "expand_dist_vs_time.png")

    def plot_tip_distance_vs_time(self, exp_id: int):
        tip_distance_data = self.exp.result_data['tip_distance'][exp_id]
        frames = range(len(tip_distance_data))
        exp_name = self.exp.experiment_names[exp_id]

        fig, ax = self._setup_plot(f"{exp_name}: Tip Distance vs. Time", "Frame Index", "Tip Distance (m)")
        ax.plot(frames, tip_distance_data, color=self.colors[8], label='Tip Distance', **self.plot_params)
        ax.legend(fontsize=self.font_sizes['legend'])
        self._save_plot(fig, exp_id, "tip_distance_vs_time.png")

    def plot_summary_for_experiment(self, exp_id: int):
        logger.info(f"Generating summary plot for experiment {exp_id}...")
        exp_name = self.exp.experiment_names[exp_id]
        
        data_keys = ['area', 'arc_length', 'area_vec', 'regression_circle_radius', 'expand_dist', 'expand_vec', 'tip_distance']
        data = {key: self.exp.result_data[key][exp_id] for key in data_keys}
        
        frames_full = range(len(data['area']))
        frames_vec = range(1, len(data['area_vec'][1:]) + 1)

        fig, axs = plt.subplots(7, 1, figsize=(12, 35))
        fig.suptitle(f"{exp_name}: Summary Plots", fontsize=self.font_sizes['title'] + 4, fontweight='bold', y=0.99)

        plot_configs = [
            {'ax_idx': 0, 'data_key': 'area', 'frames': frames_full, 'ylabel': "Contour Area (m^2)", 'title': "Contour Area", 'color_idx': 0},
            {'ax_idx': 1, 'data_key': 'arc_length', 'frames': frames_full, 'ylabel': "Arc Length (m)", 'title': "Arc Length", 'color_idx': 1},
            {'ax_idx': 2, 'data_key': 'area_vec', 'frames': frames_vec, 'data': data['area_vec'][1:], 'ylabel': "Area Change (m^2/s)", 'title': "Area Change", 'color_idx': 2},
            {'ax_idx': 3, 'data_key': 'regression_circle_radius', 'frames': frames_full, 'ylabel': "Radius (m)", 'title': "Regression Circle Radius", 'color_idx': 3},
            {'ax_idx': 6, 'data_key': 'tip_distance', 'frames': frames_full, 'ylabel': "Tip Distance (m)", 'title': "Tip Distance", 'color_idx': 8},
        ]

        for conf in plot_configs:
            ax = axs[conf['ax_idx']]
            plot_data = conf.get('data', data[conf['data_key']])
            ax.plot(conf['frames'], plot_data, color=self.colors[conf['color_idx']], label=conf['title'], **self.plot_params)
            ax.set_title(conf['title'], fontsize=self.font_sizes['title'])
            ax.set_ylabel(conf['ylabel'], fontsize=self.font_sizes['label'])
            ax.set_xlabel("Frame Index", fontsize=self.font_sizes['label'])
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
            for i in range(sanitized_data.shape[1]):
                ax.plot(frames_full, sanitized_data[:, i], color=self.colors[i+4], label=labels[i], **self.plot_params)
            ax.set_title(conf['title'], fontsize=self.font_sizes['title'])
            ax.set_ylabel(conf['ylabel'], fontsize=self.font_sizes['label'])
            ax.set_xlabel("Frame Index", fontsize=self.font_sizes['label'])
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
            fig, ax = self._setup_plot(title, "Frame Index", ylabel)

            for i, exp_id in enumerate(exp_ids):
                if key in self.exp.result_data and exp_id < len(self.exp.result_data[key]):
                    data = self.exp.result_data[key][exp_id]
                    frames = range(len(data))
                    exp_name = self.exp.experiment_names[exp_id]
                    ax.plot(frames, data, color=self.colors[i % len(self.colors)], label=exp_name, **self.plot_params)
                else:
                    logger.warning(f"Data for key '{key}' and experiment ID {exp_id} not found. Skipping.")

            ax.legend(fontsize=self.font_sizes['legend'])
            self._save_plot(fig, None, f"comparison_{key}.png")