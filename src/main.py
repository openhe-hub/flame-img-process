from config import Config
from data_manager import DataManager
from processing import exec_once
from visualizer import Visualizer


if __name__ == '__main__':
    config = Config('assets/config.toml')
    data_manager = DataManager(config)
    data_manager.load_all_experiment()
    exec_once(data_manager, 0)
    # data_manager.save_all_experiment()
    vis = Visualizer(config, data_manager)
    vis.plot_all_for_experiment(0)