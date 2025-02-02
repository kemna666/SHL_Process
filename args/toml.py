import tomllib
import argparse


class args:
    def __init__(self):
        self.get_config()
        
    def get_config(self):
        with open('./config/default.toml', 'rb') as f:
            config = tomllib.load(f)
            self.mode = config['mode']
            self.model = config['model']
            self.optim = config['optim']
            self.criterion = config['criterion']
            self.epochs = config['epochs']
            self.batch_size = config['batch_size']
            self.lr = config['lr']