"""
Author: Yanting Miao
"""

class TransformerOptim():
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        """
        :param optimizer: the optimizer that we used, in the original paper, Vaswani et al. use Adam.
        :param d_model: the embedding dimension.
        :param warmup_steps: the number of warm up training steps.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0

    def update_lr(self):
        """
        Updated learning rate. lr := d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        """
        self.n_steps += 1
        lr = self.d_model**(-0.5) * min(self.n_steps**(-0.5), self.n_steps * self.warmup_steps**(-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.update_lr()
        self.optimizer.step()
