import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.SHL import SHLDataset as SHL
from model.googlenet import GoogLeNet
from args.toml import args
'''
此处用到的参数：
1.mode：模型处于什么模式，train、test、vaild
2.model：选择模型
3.optim：选择优化器
4.lr：学习率
5.epochs：训练轮数
6.batch_size：批次大小

'''
class model:
    def __init__(self,args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_data()
        self.choose_model()
        self.choose_optim()
        self.choose_criterion()
        self.train()
        self.vaild()
        self.test()

    #加载数据
    def load_data(self):
        if self.args.dataset == 'SHL':
            self.train_dataset = SHL(mode='train',file_path=self.args.train_file_path,label_path=self.args.train_label_path)
            self.valid_dataset = SHL(mode='vaild',file_path=self.args.valid_file_path,label_path=self.args.valid_label_path)
            self.test_dataset = SHL(mode='test',file_path=self.args.test_file_path)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.args.batch_size, shuffle=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)
        #可以用其他数据集在此处套公式
        else:
            raise EnvironmentError('需要指定数据集')
    #选择加速器
    def choose_optim(self):
        if self.args.optim == 'Adam':
            self.optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'SGD':
            self.optim = optim.SGD(self.model.parameters(), lr=self.args.lr)
        #此处留位置，用于添加其他优化器
        else:
            raise EnvironmentError('没有优化器')   
    
    
    def choose_model(self):
        if self.args.model == 'GoogleNet':
            #此处需要修改参数
            self.model = GoogleNet().to(self.device)
        #以下是计划做出的模型

        else:
            raise EnvironmentError('没有模型')
    

    #选择损失函数
    def choose_criterion(self):
        if self.args.criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        #以下留空，用于添加损失函数
        else:
            raise EnvironmentError('没有损失函数')
    #开始训练
    def train(self):
        self.model.train()
        running_loss = 0.0
        for epoch in range(self.args.epochs):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs.to(self.device), targets.to(self.device)
                self.optim.zero_grad()

                if self.args.model == 'googlenet':
                    outputs, aux1, aux2 = self.model(inputs).to(self.device)
                    # 计算多损失（主损失 + 辅助损失）
                    loss_main = self.criterion(outputs, targets)
                    loss_aux1 = self.criterion(aux1, targets)
                    loss_aux2 = self.criterion(aux2, targets)
                    loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2 
                else:
                    outputs = self.model(inputs).to(self.device)
                    loss = self.criterion(outputs, targets)
                # Backward
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                if batch_idx % 10 == 9:  # 每10个batch打印一次
                    print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}')
                    running_loss = 0.0
            scheduler.step()
    def vaild(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, targets) in self.valid_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.args.model == 'googlenet':
                    # 只取主输出
                    outputs, _, _ = self.model(inputs)  
                else:
                    outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'Valid Accuracy: {acc:.2f}%')
    
    #用于测试的函数
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, targets) in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.args.model == 'googlenet':
                    # 只取主输出
                    outputs, _, _ = self.model(inputs)  
                else:
                    outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'Test Accuracy: {acc:.2f}%')



if __name__ == '__main__':
    args = args()
    model = model(args)
