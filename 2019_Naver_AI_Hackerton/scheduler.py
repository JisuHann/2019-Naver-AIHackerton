from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWithRestartsLR(_LRScheduler):
    '''
    SGDR\: Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983
    code: https://github.com/gurucharanmk/PyTorch_CosineAnnealingWithRestartsLR/blob/master/CosineAnnealingWithRestartsLR.py
    added restart_decay value to decrease lr for every restarts
    '''
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1, restart_decay=0.95):
        self.T_max = T_max
        self.T_mult = T_mult
        self.next_restart = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.last_restart = 0
        self.T_num = 0
        self.restart_decay = restart_decay
        super(CosineAnnealingWithRestartsLR,self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.Tcur = self.last_epoch - self.last_restart
        if self.Tcur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = self.last_epoch
            self.T_num += 1
        learning_rate = [(self.eta_min + ((base_lr)*self.restart_decay**self.T_num - self.eta_min) * (1 + math.cos(math.pi * self.Tcur / self.next_restart)) / 2) for base_lr in self.base_lrs]
        return learning_rate