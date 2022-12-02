import torch

def sphere_levelset_loss(model, input):
    pred = model(input)
    loss = torch.linalg.norm(pred, dim=1, keepdim=True) - 1.0
    return loss