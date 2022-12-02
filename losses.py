import torch

# Toy case: level set function of sphere with radius 1.0
def sphere_levelset_loss(model, input):
    pred, _ = model(input)
    gt = torch.linalg.norm(input, dim=1, keepdim=True) - torch.ones_like(input)
    loss = torch.mean(torch.abs(pred - gt))
    return loss