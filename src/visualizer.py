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

    def plot_area_vs_time(self, exp_id: int):
        area_data = self.exp.result_data['area'][exp_id]
        frames = range(len(area_data))
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, area_data, marker='.', linestyle='-')
        plt.xlabel("Frame Index")
        plt.ylabel("Contour Area (m^2)")
        plt.title(f"Experiment {exp_id}: Contour Area vs. Time")
        plt.grid(True)
        
        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "area_vs_time.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved area vs time plot to {save_path}")

    def plot_arc_length_vs_time(self, exp_id: int):
        arc_length_data = self.exp.result_data['arc_length'][exp_id]
        frames = range(len(arc_length_data))

        plt.figure(figsize=(10, 6))
        plt.plot(frames, arc_length_data, marker='.', linestyle='-')
        plt.xlabel("Frame Index")
        plt.ylabel("Arc Length (m)")
        plt.title(f"Experiment {exp_id}: Arc Length vs. Time")
        plt.grid(True)

        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "arc_length_vs_time.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved arc length vs time plot to {save_path}")

    def plot_area_vec_vs_time(self, exp_id: int):
        # area_vec is area change, skip the first frame which is 0
        area_vec_data = self.exp.result_data['area_vec'][exp_id][1:]
        frames = range(1, len(area_vec_data) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(frames, area_vec_data, marker='.', linestyle='-')
        plt.xlabel("Frame Index")
        plt.ylabel("Area Change (m^2/s)")
        plt.title(f"Experiment {exp_id}: Area Change vs. Time")
        plt.grid(True)

        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "area_vec_vs_time.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved area vec vs time plot to {save_path}")

    def plot_regression_circle_radius_vs_time(self, exp_id: int):
        radius_data = self.exp.result_data['regression_circle_radius'][exp_id]
        frames = range(len(radius_data))

        plt.figure(figsize=(10, 6))
        plt.plot(frames, radius_data, marker='.', linestyle='-')
        plt.xlabel("Frame Index")
        plt.ylabel("Radius (m)")
        plt.title(f"Experiment {exp_id}: Regression Circle Radius vs. Time")
        plt.grid(True)

        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "regression_circle_radius_vs_time.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved regression circle radius vs time plot to {save_path}")

    def plot_expand_vec_vs_time(self, exp_id: int):
        expand_dist_data = self.exp.result_data['expand_vec'][exp_id]
        frames = range(len(expand_dist_data))

        # Sanitize data: replace empty lists (where no contour was found) with zeros
        sanitized_expand_dist = [item if item else [0, 0, 0, 0] for item in expand_dist_data]
        expand_dist_data_np = np.array(sanitized_expand_dist)

        # Transpose the data to plot each direction
        dist_up = expand_dist_data_np[:, 0]
        dist_right = expand_dist_data_np[:, 1]
        dist_down = expand_dist_data_np[:, 2]
        dist_left = expand_dist_data_np[:, 3]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, dist_up, marker='.', linestyle='-', label='Up')
        plt.plot(frames, dist_right, marker='.', linestyle='-', label='Right')
        plt.plot(frames, dist_down, marker='.', linestyle='-', label='Down')
        plt.plot(frames, dist_left, marker='.', linestyle='-', label='Left')
        
        plt.xlabel("Frame Index")
        plt.ylabel("Velocity (m)")
        plt.title(f"Experiment {exp_id}: Expand Velocity vs. Time")
        plt.grid(True)
        plt.legend()

        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "expand_vel_vs_time.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved expand distance vs time plot to {save_path}")
    
    def plot_expand_dist_vs_time(self, exp_id: int):
        expand_dist_data = self.exp.result_data['expand_dist'][exp_id]
        frames = range(len(expand_dist_data))

        # Sanitize data: replace empty lists (where no contour was found) with zeros
        sanitized_expand_dist = [item if item else [0, 0, 0, 0] for item in expand_dist_data]
        expand_dist_data_np = np.array(sanitized_expand_dist)

        # Transpose the data to plot each direction
        dist_up = expand_dist_data_np[:, 0]
        dist_right = expand_dist_data_np[:, 1]
        dist_down = expand_dist_data_np[:, 2]
        dist_left = expand_dist_data_np[:, 3]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, dist_up, marker='.', linestyle='-', label='Up')
        plt.plot(frames, dist_right, marker='.', linestyle='-', label='Right')
        plt.plot(frames, dist_down, marker='.', linestyle='-', label='Down')
        plt.plot(frames, dist_left, marker='.', linestyle='-', label='Left')
        
        plt.xlabel("Frame Index")
        plt.ylabel("Distance (m)")
        plt.title(f"Experiment {exp_id}: Expand Distance vs. Time")
        plt.grid(True)
        plt.legend()

        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "expand_dist_vs_time.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved expand distance vs time plot to {save_path}")

    def plot_summary_for_experiment(self, exp_id: int):
        logger.info(f"Generating summary plot for experiment {exp_id}...")

        # 1. Get all the data for the experiment
        area_data = self.exp.result_data['area'][exp_id]
        arc_length_data = self.exp.result_data['arc_length'][exp_id]
        area_vec_data = self.exp.result_data['area_vec'][exp_id][1:]
        radius_data = self.exp.result_data['regression_circle_radius'][exp_id]
        expand_dist_data = self.exp.result_data['expand_dist'][exp_id]
        expand_vel_data = self.exp.result_data['expand_vec'][exp_id]
        
        frames_full = range(len(area_data))
        frames_vec = range(1, len(area_vec_data) + 1)

        # Sanitize and transpose expand_dist_data
        sanitized_expand_dist = [item if item else [0, 0, 0, 0] for item in expand_dist_data]
        expand_dist_data_np = np.array(sanitized_expand_dist)
        dist_up = expand_dist_data_np[:, 0]
        dist_right = expand_dist_data_np[:, 1]
        dist_down = expand_dist_data_np[:, 2]
        dist_left = expand_dist_data_np[:, 3]

        # Sanitize and transpose expand_dist_data
        sanitized_expand_vel = [item if item else [0, 0, 0, 0] for item in expand_vel_data]
        expand_vel_data_np = np.array(sanitized_expand_vel)
        vel_up = expand_vel_data_np[:, 0]
        vel_right = expand_vel_data_np[:, 1]
        vel_down = expand_vel_data_np[:, 2]
        vel_left = expand_vel_data_np[:, 3]

        # 2. Create a figure with 5 subplots
        fig, axs = plt.subplots(6, 1, figsize=(12, 30))
        fig.suptitle(f"Experiment {exp_id}: Summary Plots", fontsize=16)

        # Plot 1: Area vs. Time
        axs[0].plot(frames_full, area_data, marker='.', linestyle='-', color='blue')
        axs[0].set_xlabel("Frame Index")
        axs[0].set_ylabel("Contour Area (m^2)")
        axs[0].set_title("Contour Area vs. Time")
        axs[0].grid(True)

        # Plot 2: Arc Length vs. Time
        axs[1].plot(frames_full, arc_length_data, marker='.', linestyle='-', color='green')
        axs[1].set_xlabel("Frame Index")
        axs[1].set_ylabel("Arc Length (m)")
        axs[1].set_title("Arc Length vs. Time")
        axs[1].grid(True)

        # Plot 3: Area Change vs. Time
        axs[2].plot(frames_vec, area_vec_data, marker='.', linestyle='-', color='red')
        axs[2].set_xlabel("Frame Index")
        axs[2].set_ylabel("Area Change (m^2/s)")
        axs[2].set_title("Area Change vs. Time")
        axs[2].grid(True)

        # Plot 4: Regression Circle Radius vs. Time
        axs[3].plot(frames_full, radius_data, marker='.', linestyle='-', color='purple')
        axs[3].set_xlabel("Frame Index")
        axs[3].set_ylabel("Radius (m)")
        axs[3].set_title("Regression Circle Radius vs. Time")
        axs[3].grid(True)

        # Plot 5: Expand Distance vs. Time
        axs[4].plot(frames_full, dist_up, marker='.', linestyle='-', label='Up')
        axs[4].plot(frames_full, dist_right, marker='.', linestyle='-', label='Right')
        axs[4].plot(frames_full, dist_down, marker='.', linestyle='-', label='Down')
        axs[4].plot(frames_full, dist_left, marker='.', linestyle='-', label='Left')
        axs[4].set_xlabel("Frame Index")
        axs[4].set_ylabel("Distance (m)")
        axs[4].set_title("Expand Distance vs. Time")
        axs[4].grid(True)
        axs[4].legend()

        # Plot 5: Expand Distance vs. Time
        axs[5].plot(frames_full, vel_up, marker='.', linestyle='-', label='Up')
        axs[5].plot(frames_full, vel_right, marker='.', linestyle='-', label='Right')
        axs[5].plot(frames_full, vel_down, marker='.', linestyle='-', label='Down')
        axs[5].plot(frames_full, vel_left, marker='.', linestyle='-', label='Left')
        axs[5].set_xlabel("Frame Index")
        axs[5].set_ylabel("Velocity (m)")
        axs[5].set_title("Expand Velocity vs. Time")
        axs[5].grid(True)
        axs[5].legend()

        # 3. Adjust layout and save the combined plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        plot_dir = os.path.join(self.output_dir, f"experiment_{exp_id}", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "summary_plot.png")
        plt.savefig(save_path)
        plt.close(fig)
        
        logger.success(f"Finished generating summary plot for experiment {exp_id}, saved to {save_path}")

    def plot_all_for_experiment(self, exp_id: int, plot_summary: bool = True, plot_single: bool = True):
        logger.info(f"Generating all plots for experiment {exp_id}...")
        if plot_single:
            self.plot_area_vs_time(exp_id)
            self.plot_arc_length_vs_time(exp_id)
            self.plot_area_vec_vs_time(exp_id)
            self.plot_regression_circle_radius_vs_time(exp_id)
            self.plot_expand_dist_vs_time(exp_id)
            self.plot_expand_vec_vs_time(exp_id)
        if plot_summary:
            self.plot_summary_for_experiment(exp_id)
        logger.success(f"Finished generating plots for experiment {exp_id}")

    def plot_all(self, plot_summary: bool = True, plot_single: bool = True):
        logger.info("Generating plots for all experiments...")
        for exp_id in range(len(self.exp.experiment_imgs)):
            self.plot_all_for_experiment(exp_id, plot_summary, plot_single)
        logger.success("Finished generating all plots.")
