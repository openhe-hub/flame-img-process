from config import Config
from data_manager import Dataloader
from processing import exec_once


if __name__ == '__main__':
    config = Config('assets/config.toml')
    dataloader = Dataloader(config)
    dataloader.load_all_experiment()
    exec_once(dataloader, 0)
    dataloader.save_all_experiment()