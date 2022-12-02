import numpy as np
import torch
from torch.utils.data import Dataset


class RectSampler(Dataset):
    def __init__(self, sample_num):
        super(RectSampler, self).__init__()
        self.sample_num = sample_num

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.zeros(self.sample_num, 2).uniform_(-2., 2.)