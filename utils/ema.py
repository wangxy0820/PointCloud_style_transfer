# utils/ema.py

import torch
from collections import OrderedDict
from typing import Iterator

class ExponentialMovingAverage:
    """
    指数移动平均 (EMA) 工具。
    用于平滑模型参数，通常可以在训练后期获得更好的、更稳定的模型。
    在验证和推理时，我们会使用EMA参数而不是原始的模型参数。
    """
    def __init__(self, parameters: Iterator[torch.nn.Parameter], decay: float = 0.995):
        """
        Args:
            parameters: an iterator of the model's parameters.
            decay: the decay factor for EMA.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        
        # 将参数转换为列表，确保一致性
        self.parameters = list(parameters)
        self.decay = decay
        self.collected_params = []
        
        # 初始化shadow parameters - 只为需要梯度的参数创建shadow
        self.shadow_params = []
        self.param_names = []  # 添加参数名跟踪，便于调试
        
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.requires_grad:
                    self.shadow_params.append(param.clone().detach())
                    self.param_names.append(f"param_{i}")
        
        print(f"EMA initialized with {len(self.shadow_params)} shadow parameters out of {len(self.parameters)} total parameters")

    def update(self):
        """
        在每个训练步骤后调用此方法，以更新EMA参数。
        """
        if len(self.shadow_params) == 0:
            return
            
        with torch.no_grad():
            shadow_idx = 0
            for param in self.parameters:
                if param.requires_grad:
                    if shadow_idx < len(self.shadow_params):
                        # 使用in-place操作提高效率
                        self.shadow_params[shadow_idx].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                    shadow_idx += 1

    def apply_shadow(self):
        """
        将EMA参数（shadow_params）复制到模型中。
        通常在验证或推理之前调用。
        """
        if len(self.shadow_params) == 0:
            return
            
        self.collected_params = []
        shadow_idx = 0
        
        with torch.no_grad():
            for param in self.parameters:
                if param.requires_grad:
                    if shadow_idx < len(self.shadow_params):
                        self.collected_params.append(param.data.clone())
                        param.data.copy_(self.shadow_params[shadow_idx])
                    shadow_idx += 1

    def restore(self):
        """
        将原始参数恢复到模型中。
        在验证或推理后调用。
        """
        if len(self.collected_params) == 0:
            return
            
        collected_idx = 0
        with torch.no_grad():
            for param in self.parameters:
                if param.requires_grad:
                    if collected_idx < len(self.collected_params):
                        param.data.copy_(self.collected_params[collected_idx])
                    collected_idx += 1
        self.collected_params = []

    def state_dict(self) -> dict:
        """
        返回EMA状态字典，用于保存。
        """
        return {
            'decay': self.decay,
            'shadow_params': [p.clone().detach().cpu() for p in self.shadow_params]
        }

    def load_state_dict(self, state_dict: dict):
        """
        加载EMA状态字典。
        """
        if 'shadow_params' not in state_dict:
            raise KeyError("Invalid EMA state_dict: missing 'shadow_params'")
            
        self.decay = state_dict['decay']
        loaded_shadow_params = state_dict['shadow_params']
        
        # 检查参数数量
        expected_params = len([p for p in self.parameters if p.requires_grad])
        loaded_param_count = len(loaded_shadow_params)
        
        if loaded_param_count != expected_params:
            print(f"Warning: Loaded EMA params count ({loaded_param_count}) != expected ({expected_params})")
            print("This may indicate model architecture changes between training and inference")
            
            # 尝试部分加载兼容的参数
            min_count = min(loaded_param_count, expected_params, len(self.shadow_params))
            if min_count > 0:
                print(f"Loading first {min_count} compatible EMA parameters")
                for i in range(min_count):
                    if i < len(self.shadow_params):
                        device = self.parameters[0].device if len(self.parameters) > 0 else 'cpu'
                        self.shadow_params[i] = loaded_shadow_params[i].to(device)
                return
            else:
                print("No compatible parameters found, reinitializing EMA")
                self.__init__(self.parameters, self.decay)
                return
        
        # 完全匹配的情况：正常加载所有参数
        self.shadow_params = []
        shadow_idx = 0
        
        for param in self.parameters:
            if param.requires_grad:
                if shadow_idx < len(loaded_shadow_params):
                    shadow_param = loaded_shadow_params[shadow_idx].to(param.device)
                    
                    # 验证形状匹配
                    if shadow_param.shape != param.shape:
                        print(f"Warning: EMA parameter shape mismatch at index {shadow_idx}: "
                              f"loaded {shadow_param.shape} vs expected {param.shape}")
                        # 如果形状不匹配，使用当前参数重新初始化
                        shadow_param = param.clone().detach()
                    
                    self.shadow_params.append(shadow_param)
                shadow_idx += 1
        
        print(f"EMA state loaded: {len(self.shadow_params)} parameters restored")