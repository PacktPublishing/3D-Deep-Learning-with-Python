import torch

from pytorch3d.transforms.so3 import (so3_exp_map,
                                      so3_log_map,
                                      hat_inv, hat)

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
else:
    device = torch.device("cpu") 
    print("WARNING: CPU only, this will be slow!")

log_rot = torch.zeros([4, 3], device = device)
log_rot[0, 0] = 0.001 
log_rot[0, 1] = 0.0001 
log_rot[0, 2] = 0.0002 

log_rot[1, 0] = 0.0001 
log_rot[1, 1] = 0.001 
log_rot[1, 2] = 0.0002 

log_rot[2, 0] = 0.0001 
log_rot[2, 1] = 0.0002 
log_rot[2, 2] = 0.001 

log_rot[3, 0] = 0.001 
log_rot[3, 1] = 0.002 
log_rot[3, 2] = 0.003

log_rot_hat = hat(log_rot)
print('log_rot_hat shape = ', log_rot_hat.shape)
print('log_rot_hat = ', log_rot_hat)

log_rot_copy = hat_inv(log_rot_hat)
print('log_rot_copy shape = ', log_rot_copy.shape)
print('log_rot_copy = ', log_rot_copy)

rotation_matrices = so3_exp_map(log_rot)
print('rotation_matrices = ', rotation_matrices)

log_rot_again = so3_log_map(rotation_matrices)
print('log_rot_again = ', log_rot_again)
