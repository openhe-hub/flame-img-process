import toml
from loguru import logger

class Config:
    def __init__(self, config_path):
        self.config = toml.load(config_path)
        logger.success("config loaded")
    
    def get_config(self):
        return self.config