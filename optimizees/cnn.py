import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from .base import BaseOptimizee

torch.set_default_dtype(torch.float32)

class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(3 * 7 * 7, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        x = x.view(-1, 3 * 7 * 7)
        x = self.fc(x)
        return x

class MnistCNN(BaseOptimizee):
    def __init__(
        self,
        batch_size: int,
        test_batch_size: int = 1000,
        device='cpu',
        **options
    ) -> None:
        """
        小型CNN在MNIST数据集上的训练优化问题，使用全量数据计算目标函数和梯度
        
        参数：
            batch_size: 训练批次大小 (用于数据加载，但目标函数使用全量数据)
            test_batch_size: 测试批次大小
            device: 计算设备
        """
        self.device = device
        self.vars = dict()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        
        # 加载MNIST数据集
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        
        test_dataset = datasets.MNIST('./data', train=False, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        
        # 创建模型
        self.model = TinyCNN().to(device)
        
        # 加载全部训练数据到内存 (用于目标函数计算)
        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 获取全部训练数据
        for data, target in self.train_data_loader:
            self.full_train_data = data.to(device)
            self.full_train_target = target.to(device)
            break
        
        # 加载全部测试数据到内存
        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=False)
        
        # 获取全部测试数据
        for data, target in self.test_data_loader:
            self.full_test_data = data.to(device)
            self.full_test_target = target.to(device)
            break
        
        print(f"Loaded {len(self.full_train_data)} training samples")
        print(f"Training data shape: {self.full_train_data.shape}")
        print(f"Loaded {len(self.full_test_data)} test samples")
        print(f"Test data shape: {self.full_test_data.shape}")
        
        # 初始化模型参数作为优化变量
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数总数: {param_count}")
        
        # 将模型参数展平为一个向量，作为优化变量
        self.X = self.flatten_params().unsqueeze(0).unsqueeze(-1)  # [batch_size, dim, 1]
        self.set_var('X', self.X)
        self.set_var('y', self.X)
        

    def get_var(self, var_name):
        return self.vars[var_name]

    def set_var(self, var_name, var_value):
        self.vars[var_name] = var_value

    def detach_vars(self):
        for var in self.vars.values():
            var.detach()

    @property
    def X(self):
        return self.get_var('X')

    @X.setter
    def X(self, value):
        self.set_var('X', value)
        
    def flatten_params(self):
        """将模型参数展平为一个向量"""
        params = []
        for p in self.model.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params, 0).detach().clone()
    
    def load_params_from_vector(self, x):
        """从展平的向量加载参数到模型中"""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(x[offset:offset + numel].view(p.shape))
            offset += numel
    
    def generate(self, batch_size: int):
        """生成新的批次数据（保持接口兼容性，但目标函数使用全量数据）"""
        return {'data': self.full_train_data, 'target': self.full_train_target}

    def objective(self, inputs: dict = None, compute_grad: bool = False):
        """计算目标函数值（使用全量MNIST训练数据的交叉熵损失）"""
        if inputs is None:
            inputs = {}
        
        X = inputs.get('X', self.X)
        # 总是使用全量训练数据
        data = self.full_train_data
        target = self.full_train_target
        
        with torch.set_grad_enabled(compute_grad):
            # 将优化变量加载到模型中
            if X.dim() == 3:  # [batch_size, dim, 1]
                param_vec = X.squeeze(-1).squeeze(0)
            elif X.dim() == 2:  # [batch_size, dim]
                param_vec = X.squeeze(0)
            else:  # [dim]
                param_vec = X
                
            self.load_params_from_vector(param_vec)
            
            # 前向传播计算损失（全量数据）
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            return loss
    
    def bp_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """使用PyTorch的反向传播机制计算梯度（使用全量MNIST训练数据）"""
        if inputs is None:
            inputs = {}
        
        # 获取当前X，确保包含正确维度
        X = inputs.get('X', self.X).detach().clone()
        
        # 确保X具有正确的形状 [batch_size, dim, 1]
        if X.dim() == 2:  # 如果是 [batch_size, dim]
            X = X.unsqueeze(-1)
        elif X.dim() == 1:  # 如果是 [dim]
            X = X.unsqueeze(0).unsqueeze(-1)  # 添加batch和最后一个维度
        
        # 提取参数向量
        param_vector = X.squeeze(-1).squeeze(0)  # [dim]
        param_vector = param_vector.clone()
        param_vector.requires_grad_(True)
        
        # 加载参数向量到模型
        self.load_params_from_vector(param_vector)
        
        # 使用全量训练数据
        data = self.full_train_data
        target = self.full_train_target
        
        # 前向传播（全量数据）
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        
        # 直接计算梯度
        loss.backward(retain_graph=True)
        
        # 收集所有参数的梯度
        all_grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                all_grads.append(p.grad.view(-1))
            else:
                all_grads.append(torch.zeros_like(p.view(-1)))
        
        # 拼接所有梯度
        grad = torch.cat(all_grads)
        
        # 清除模型中的梯度
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        # 添加batch和最后一个维度
        grad = grad.unsqueeze(0).unsqueeze(-1)  # [1, dim, 1]
        
        return grad
    
    def smooth_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """计算光滑部分的梯度（使用全量数据）"""
        return self.bp_grad(inputs, compute_grad, **kwargs)
    
    def cuda(self):
        """将模型和数据移至CUDA设备"""
        self.device = 'cuda'
        self.model.cuda()
        self.full_train_data = self.full_train_data.cuda()
        self.full_train_target = self.full_train_target.cuda()
        self.full_test_data = self.full_test_data.cuda()
        self.full_test_target = self.full_test_target.cuda()
        for k, v in self.vars.items():
            if isinstance(v, torch.Tensor):
                self.vars[k] = v.cuda()
        return self
    
    def evaluate(self):
        """在全量测试集上评估模型性能"""
        self.model.eval()
        with torch.no_grad():
            # 使用全量测试数据
            data, target = self.full_test_data, self.full_test_target
            output = self.model(data)
            
            # 计算总损失
            test_loss = F.cross_entropy(output, target, reduction='mean').item()
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(target)
        
        return test_loss, accuracy
    
    def get_grad(
        self,
        grad_method: str,
        inputs: dict = None,
        compute_grad: bool = False,
        **kwargs
    ):
        """
        计算指定类型的梯度，由`grad_method`指定。
        `grad_method`可以从['smooth_grad', 'bp_grad']中选择。
        """
        grad_func = getattr(self, grad_method, None)
        if grad_func is None:
            raise RuntimeError(f'无效的梯度方法: {grad_method}')
        return grad_func(inputs, compute_grad, **kwargs)
    
    def grad_lipschitz(self):
        """计算梯度Lipschitz常数"""
        return torch.ones(self.X.size(0)).to(device=self.X.device)

    def objective_batch(self, inputs: dict = None, compute_grad: bool = False):
        """计算批次目标函数值（使用全量数据）"""
        if inputs is None:
            inputs = {}
        
        # 计算损失并确保形状正确
        loss = self.objective(inputs, compute_grad)
        return loss.unsqueeze(0)  # [batch_size]

    def prox(self, inputs: dict, compute_grad: bool = False):
        X = inputs['X']
        with torch.set_grad_enabled(compute_grad):
            return X

    def subgrad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """计算子梯度（使用全量数据）"""
        return self.bp_grad(inputs, compute_grad, **kwargs)
    

