import os

import numpy as np
import torch
from networks import Siren

def main():
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Let's use CUDA")
    else:
        device = torch.device('cpu')


if __name__ == '__main__':
    main()