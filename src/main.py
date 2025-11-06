from config import Config
from data_manager import DataManager
from processing import exec_once
from visualizer import Visualizer


if __name__ == '__main__':
    config = Config('assets/config.toml')
    data_manager = DataManager(config)
    data_manager.load_all_experiment(end_idx=3)

    num_experiments = len(data_manager.experiment.experiment_imgs)
    for i in range(num_experiments):
        exec_once(data_manager, i)
        
    data_manager.save_all_experiment()
    # vis = Visualizer(config, data_manager)
    # vis.plot_all()