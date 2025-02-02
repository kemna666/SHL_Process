import tomllib
import argparse


class args:
    def __init__(self):
        self.get_config()
        
    def get_config(self):
        with open('./config/default.toml', 'rb') as f:
            config = tomllib.load(f)
            self.model = config['model']['model']
            self.optim = config['config']['optim']
            self.criterion = config['config']['criterion']
            self.epochs = config['config']['epochs']
            self.batch_size = config['dataset']['batch_size']
            self.lr = config['config']['lr']
            self.dataset = config['dataset']['dataset']
            self.train_label_path = config['dataset']['train_label_path']
            self.train_file_path = config['dataset']['train_file_path']
            self.test_file_path = config['dataset']['test_file_path']
            self.valid_label_path = config['dataset']['valid_label_path']
            self.valid_file_path = config['dataset']['valid_file_path']
