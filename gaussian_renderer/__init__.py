#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import trimesh


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def norm_gauss(m, sigma, t):
    log = ((m - t)**2 / sigma**2) / -2
    return torch.exp(log)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, interp=1, interp_idx=0, modify_func=None, mask_tensor=None, x_grid=None, y_grid=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    viewpoint_camera.camera_center = viewpoint_camera.camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        antialiasing=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    _xyz = pc.get_xyz
    means3D = _xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    time_func = pc.get_time

    camera_time = viewpoint_camera.time

    time = 0 + torch.sum(time_func[:camera_time]).repeat(means3D.shape[0],1)
    time_next = 0 + torch.sum(time_func[:camera_time+1]).repeat(means3D.shape[0],1)

    time = time + (time_next - time) * interp_idx / interp

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        rotations = pc.get_rotation

    # shape: [num_gaussians, 2 * polynomial_degree] -> [num_gaussians, 2] x polynomial_degree
    poly_weights = torch.chunk(pc._w1, chunks=pc.polynomial_degree, dim=-1)

    # means3D = means3D[:, [0, -1]] + pc._w1 * (pc.get_m - time[0])
    means3D = means3D[:, [0, -1]]
    center_gaussians = pc.get_m - time[0]
    for i, poly_weight in enumerate(poly_weights):
        means3D = means3D + poly_weight * (center_gaussians ** (i+1))


    means3D = torch.cat([means3D[:, 0].unsqueeze(1),
                        torch.zeros(means3D[:, 0].shape).unsqueeze(1).cuda(),
                        means3D[:, -1].unsqueeze(1)]
                        , dim=1)
    
    delta = norm_gauss(pc.get_m.squeeze(), pc.get_sigma.squeeze(), time[0]).unsqueeze(-1)
    scales = delta * pc.get_scaling 

    mask1 = (delta > 0.01).all(dim=1)
    s = scales[:,[0,-1]]
    mask2 = (s > 0.0001).all(dim=1)
    mask = torch.logical_and(mask1, mask2)

    if mask_tensor is not None:
        # if torch.count_nonzero(mask_tensor) != 0:
            # print(f"🚨 WARNING: Mask is not empty")

        homogeneous_coords = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)  # Convert to homogeneous coords
        projected = torch.matmul(viewpoint_camera.full_proj_transform, homogeneous_coords.T).T  # Apply projection
        projected_means2D = projected[:, :2] / projected[:, 3:4]  # Divide by w to normalize
        projected_means2D = projected_means2D.clamp(-1, 1)  # Clamp to valid range
    

        
        x_gauss = projected_means2D[:, 0]  # X-coordinates
        y_gauss = projected_means2D[:, 1]  # Y-coordinates

        # ✅ Ensure sampling grid is [1, 1, num_gaussians, 2] (no Z-dimension)
        sampling_grid = torch.stack([x_gauss, y_gauss], dim=-1).unsqueeze(0).unsqueeze(0)

        # ✅ Ensure mask_tensor is [1, 1, H, W] (remove depth if necessary)
        mask_tensor = mask_tensor.unsqueeze(1)

        if viewpoint_camera.time == 325:
            print(f"🟢 Mask Tensor shape: {mask_tensor.shape}")  # Expect [1, 1, H, W]
            print(f"🟢 Mask Tensor min/max: {mask_tensor.min().item()} → {mask_tensor.max().item()}")  # Expect non-zero if valid
            
            print(f"🟢 Sampling Grid shape: {sampling_grid.shape}")  # Expect [1, 1, num_gaussians, 2]
            print(f"🟢 Sampling Grid X range: {sampling_grid[..., 0].min().item()} → {sampling_grid[..., 0].max().item()}")
            print(f"🟢 Sampling Grid Y range: {sampling_grid[..., 1].min().item()} → {sampling_grid[..., 1].max().item()}")



        # ✅ Scale from [-1,1] to [0,1] ONLY for grid_sample()
        sampling_grid = (sampling_grid + 1) / 2  # Shift range from [-1,1] → [0,1]

        # ✅ Sample mask at Gaussian positions
        mask_values = torch.nn.functional.grid_sample(
            mask_tensor,  # Mask tensor: [1, 1, H, W]
            sampling_grid,  # Sampling grid: [1, 1, num_gaussians, 2]
            mode="nearest", align_corners=True
        ).squeeze()
        if viewpoint_camera.time == 325:
            print("✅ Sampled mask values min/max:", mask_values.min().item(), mask_values.max().item())
            print("Sampled mask values (first 10):", mask_values[:10])


        # ✅ Mask out Gaussians that fall outside the valid region
        mask_filter = mask_values > 0  # Keep only valid points
        if viewpoint_camera.time == 325:
            print("Before Masking: #Gaussians", mask.sum().item())  
            print("Mask Filtered: #Gaussians", mask_filter.sum().item())  

        mask = torch.logical_and(mask, mask_filter)  # Apply to existing mask


    if modify_func != None:
        means3D, scales, rotations = modify_func(means3D, scales, rotations, time[0])


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
   
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, _ = rasterizer(
        means3D = means3D[mask],
        means2D = means2D[mask],
        shs = None,
        colors_precomp = colors_precomp[mask],
        opacities = opacity[mask],
        scales = scales[mask],
        rotations = rotations[mask],
        cov3D_precomp = cov3D_precomp)
    
    radii_full = torch.zeros(means3D.shape[0], dtype=radii.dtype, requires_grad=False,
                                              device=bg_color.device)
    radii_full[mask] = radii
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_full > 0,
            "radii": radii_full}
