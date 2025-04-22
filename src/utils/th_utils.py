import torch
import torch.optim as optim
from torch import nn

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=gain)
        
class ReciprocalScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, lr_values, last_epoch=-1):
        
        self.milestones = milestones
        self.lr_values = lr_values
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        根据当前 epoch 返回学习率
        """
        for i, milestone in enumerate(self.milestones):
            if self.last_epoch < milestone:
                return [self.lr_values[i] for _ in self.base_lrs]
        return [self.lr_values[-1] for _ in self.base_lrs]