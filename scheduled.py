import random
import numpy as np


class ScheduledSampler(object):
    def __init__(
        self,
        mode='constant',
        method='average',
        noise=False,
        ratio_init=1.0,
        ratio_final=0.5,
        start_decay=50000,
        decay_steps=50000,
        allow_bp=False
    ):
        """
        A scheduled sampler that carries out scheduled sampling (ss)

        Args:
            mode (str): ss ratio update mode, 'constant': constant ratio, 'scheduled': scheduled decay
            method (str): method for generating the memory, 'average': weighted average, 'switch': random sampling
            noise (bool): whether adding a constant noise to the memory
            ratio_init (float): initial ss ratio, for both 'constant' and 'scheduled'
            ratio_final (float): final ss ratio, only for 'scheduled'
            start_decay (int): start decay step, only for 'scheduled'
            decay_steps (int): number of decay steps, only for 'scheduled'
            allow_bp (bool): whether back-propagating the error through the predicted output
        """
        self.mode = mode
        self.method = method
        self.noise = noise
        self.ratio_init = ratio_init
        self.ratio_final = ratio_final
        self.start_decay = start_decay
        self.decay_steps = decay_steps
        self.alpha = self.ratio_final / self.ratio_init

        self.allow_bp = allow_bp
        self.ratio = self.ratio_init

    def update_ratio(self, current_step):
        if self.mode == 'scheduled' and current_step >= self.start_decay:
            self.ratio = self._cosine_decay(current_step)

    def _cosine_decay(self, current_step):
        decay_step_count = min(current_step - self.start_decay, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step_count / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        decayed_ratio = self.ratio_init * decayed
        return decayed_ratio

    def get_new_memory(self, memory_gt, memory_pred, noise):
        assert 0.0 <= self.ratio <= 1.0, "Ratio should be in range [0.0, 1.0]"
        if not self.allow_bp:
            memory_pred = memory_pred.detach()

        if self.ratio == 1.0:
            new_memory = memory_gt
        elif self.method == "average":
            new_memory = memory_gt * self.ratio + memory_pred * (1.0 - self.ratio)
        elif self.method == "switch":
            new_memory = memory_gt if random.random() <= self.ratio else memory_pred
        else:
            raise ValueError("Unknown ss method, should be average or switch")

        return new_memory + noise if self.noise else new_memory
