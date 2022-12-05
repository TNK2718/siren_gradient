import torch

# Toy case: level set function of sphere with radius 0.8
def sphere_levelset_loss(model, input):
    pred, _ = model(input)
    gt_norm = torch.linalg.norm(input, dim=1, keepdim=True)
    gt = gt_norm - 0.8 * torch.ones_like(gt_norm) # C(x, y) = sqrt(x^2+y^2) - 0.8
    loss = torch.mean(torch.abs(pred - gt.detach())) # L = |NN(x,y) - C(x,y)|
    return loss

# Debug
def plane_loss(model, input):
    pred, _ = model(input)
    return torch.mean(torch.abs(pred))