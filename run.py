import os
import argparse

import numpy as np
import torch
from networks import Siren
from dataloader import RectSampler
from torch.utils.data import Subset, DataLoader
from losses import sphere_levelset_loss, plane_loss
import matplotlib.pyplot as plt

# Copyright (c) 2020 Vincent Sitzmann
# Released under the MIT license
# https://github.com/vsitzmann/siren/blob/master/LICENSE
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def main():
    parser = argparse.ArgumentParser(description="SIREN gradient eval test")
    base_ops = parser.add_argument_group("base")
    base_ops.add_argument('--hidden_features', type=int, default=32)
    base_ops.add_argument('--hidden_layers', type=int, default=10)
    base_ops.add_argument('--samples', type=int, default=10000)
    base_ops.add_argument('--batch_size', type=int, default=1024)
    base_ops.add_argument('--lr', type=float, default=1.0e-4)
    base_ops.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Let's use CUDA")
    else:
        device = torch.device('cpu')

    model = Siren(2, args.hidden_features, args.hidden_layers, 1)
    model.to(device)

    sampler = RectSampler(args.samples // args.batch_size)
    
    train_loader = DataLoader(sampler, batch_size=args.batch_size, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs

    for epoch in range(epochs):
        # training
        model.train()
        avg_train_loss = 0.0
        for data in train_loader:
            input = data.to(device)
            # train_loss = sphere_levelset_loss(model, input)
            train_loss = plane_loss(model, input)
            # update
            optim.zero_grad(set_to_none=True)
            train_loss.backward()
            optim.step()

            avg_train_loss += train_loss.item() * args.batch_size

        avg_train_loss /= args.samples
        print("epoch={:d}, loss={:f}".format(epoch, avg_train_loss))

    # Plot predicted level set
    model.eval()
    
    sidelen = 1024
    grid = get_mgrid(sidelen) * 2

    grid = grid.to(device)
    levels, _ = model(grid)

    X = grid[:, 0]
    Y = grid[:, 1]
    X = torch.reshape(X, (sidelen, sidelen))
    Y = torch.reshape(Y, (sidelen, sidelen))
    levels_grid = torch.reshape(levels, (sidelen, sidelen))

    X = X.to('cpu').detach().numpy().copy()
    Y = Y.to('cpu').detach().numpy().copy()
    levels_grid = levels_grid.to('cpu').detach().numpy().copy()

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(121)
    ax.contourf(X, Y, levels_grid)
    plt.savefig('level_pred.png')
    plt.show()

    # 3D グラフを作成する。
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, levels_grid)
    plt.show()

    # Plot gradient of predicted level set
    grad = torch.autograd.grad(levels, [grid], create_graph=True)[0]
    
    fig = plt.figure(figsize=(19.2, 10.8))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    ax.quiver(grid[:, 0], grid[:, 1], grad[:, 0], grad[:, 1])
    plt.savefig('auto_grad.png')
    plt.show()

    # TODO: Evaluate gradient without autograd, using equation (7) in supplement Sec. 2 

    


if __name__ == '__main__':
    main()
