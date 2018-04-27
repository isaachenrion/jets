import torch

def polar_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.tensor(x, y, z, device=r.device)

def joint_angles_to_cartesian(joint_angles_tensor):
    bs, n, d = joint_angles_tensor.size()
    assert d == 3

    r = joint_angles_tensor[:,:,0]
    theta = joint_angles_tensor[:,:,0]
    phi = joint_angles_tensor[:,:,0]

    relative_cartesian = polar_to_cartesian(r, theta, phi)

    absolute_cartesian = relative_cartesian.cumsum(dim=1)
    return absolute_cartesian

    #anchor = torch.zeros_like(joint_angles_tensor) # put first joint at (0,0,0)
    #anchor[:,1,0] = 1 # put second joint at (1,0,0)
