import torch

def polar_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z

def cartesian_to_polar(x, y, z):
    r = (x ** 2 + y ** 2 + z ** 2).sqrt()
    theta = torch.acos(z / r)
    phi = torch.atan2(y,torch.max(x, torch.tensor(1e-10)))
    return r, theta, phi
