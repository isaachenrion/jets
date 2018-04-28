import torch
import math

def polar_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z

def cartesian_to_polar(x, y, z):
    r = (x ** 2 + y ** 2 + z ** 2).sqrt()
    theta = torch.acos(z / r)
    phi = (torch.atan2(y,x) + 2 * math.pi).fmod(2*math.pi)
    return r, theta, phi

def joint_angles_to_cartesian(joint_angles_tensor):
    bs, n, d = joint_angles_tensor.size()
    assert d == 3

    r = joint_angles_tensor[:,:,0]
    theta = joint_angles_tensor[:,:,1]
    phi =  joint_angles_tensor[:,:,2]

    relative_cartesian = torch.stack(polar_to_cartesian(r, theta, phi), -1)
    absolute_cartesian = relative_cartesian.cumsum(dim=1)

    return absolute_cartesian

def robust_equality(a, b):
    return torch.prod((a - b).abs() < 1e-3)

def all_tests(n=1000):
    test_polar_to_cartesian(n)
    test_cartesian_to_polar(n)
    test_joint_angles(n)

def test_polar_to_cartesian(n):
    r = torch.rand(0,1,n) * 1000
    theta = torch.rand(0,1,n) * math.pi
    phi = torch.rand(0,1,n) * 2 * math.pi
    x, y, z = polar_to_cartesian(r, theta, phi)
    r_, theta_, phi_ = cartesian_to_polar(x, y, z)
    success = test_polar_equality(r, theta, phi, r_, theta_, phi_)

def test_cartesian_to_polar(n):
    x = torch.rand(0,1,n) * 1000
    y = torch.rand(0,1,n) * 1000
    z = torch.rand(0,1,n) * 1000
    r, theta, phi = cartesian_to_polar(x, y, z)
    x_, y_, z_ = polar_to_cartesian(r, theta, phi)
    success = test_cartesian_equality(x, y, z, x_, y_, z_)

def test_joint_angles(n):
    coords = 2*torch.rand(n,3) - 1
    path = coords.cumsum(0)

    r,t,p = cartesian_to_polar(coords[:,0], coords[:,1], coords[:,2])
    joint_angles = torch.stack([r,t,p], 1).unsqueeze(0)
    absolute_cartesian = joint_angles_to_cartesian(joint_angles)
    try:
        assert robust_equality(absolute_cartesian, path)
    except AssertionError:
        print("Failed: joint angles test")
        print(absolute_cartesian)
        print(path)

    print("Passed: joint angles test")

def test_polar_equality(r, theta, phi, r_, theta_, phi_):
    success = True
    try:
        assert robust_equality(r, r_)
        #print("Success: r")
    except AssertionError:
        print("FAIL: r = {} but computed r = {}".format(r, r_))
        success = False

    try:
        assert robust_equality(theta, theta_)
        #print("Success: theta")
    except AssertionError:
        print("FAIL: theta = {} but computed theta = {}".format(theta, theta_))
        success = False

    try:
        assert robust_equality(phi, phi_)
        #print("Success: phi")
    except AssertionError:
        print("FAIL: phi = {} but computed phi = {}".format(phi, phi_))
        success = False

    if success:
        print("Passed: polar equality test")
    return success

def test_cartesian_equality(x, y, z, x_, y_, z_):
    success = True
    try:
        assert robust_equality(x, x_)
        #print("Success: x")
    except AssertionError:
        print("FAIL: x = {} but computed x = {}".format(x, x_))
        success = False

    try:
        assert robust_equality(y, y_)
        #print("Success: y")
    except AssertionError:
        print("FAIL: y = {} but computed y = {}".format(y, y_))
        success = False
    try:
        assert robust_equality(z, z_)
        #print("Success: z")
    except AssertionError:
        print("FAIL: z = {} but computed z = {}".format(z, z_))
        success = False

    if success:
        print("Passed: cartesian equality test")
    return success
