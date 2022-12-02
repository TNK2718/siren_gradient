import torch

# Toy case: level set function of sphere with radius 1.0
def sphere_levelset_loss(model, input):
    pred, _ = model(input)
    gt_norm = torch.linalg.norm(input, dim=1, keepdim=True)
    gt = gt_norm - torch.ones_like(gt_norm)
    loss = torch.mean(torch.abs(pred - gt))
    return loss

# Debug
def plane_loss(model, input):
    pred, _ = model(input)
    return torch.mean(torch.abs(pred - torch.ones_like(pred)))