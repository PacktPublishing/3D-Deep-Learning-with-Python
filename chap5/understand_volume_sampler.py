import torch
from pytorch3d.structures import Volumes
from pytorch3d.renderer.implicit.renderer import VolumeSampler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

checkpoint = torch.load('ray_sampling.pt')
ray_bundle = checkpoint.get('ray_bundle')

batch_size = 10
densities = torch.zeros([batch_size, 1, 64, 64, 64]).to(device)
colors = torch.zeros(batch_size, 3, 64, 64, 64).to(device)
voxel_size = 0.1

volumes = Volumes(
    densities=densities,
    features=colors,
    voxel_size=voxel_size
)

volume_sampler = VolumeSampler(volumes = volumes, sample_mode = "bilinear")
rays_densities, rays_features = volume_sampler(ray_bundle)

print('rays_densities shape = ', rays_densities.shape)
print('rays_features shape = ', rays_features.shape)

torch.save({
    'rays_densities': rays_densities,
    'rays_features': rays_features
}, 'volume_sampling.pt')