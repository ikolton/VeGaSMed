import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender
from PIL import Image
import torchvision.transforms as T


def modify_func(means3D: torch.Tensor, # num_gauss x 3, means3D[:,1] = 0
                scales: torch.Tensor, # num_gauss x 3, scales[:,1] = eps
                rotations: torch.Tensor, # # num_gauss x 4, 3D quaternions of 2D rotations
                time: float):
    return means3D, scales, rotations

def load_mask(mask_folder, frame_idx, H, W):
    """Load a binary mask image for a specific frame and convert it to normalized coordinates (-1 to 1)."""
    mask_path = os.path.join(mask_folder, f"{frame_idx:04d}.png")  # Assuming masks are named 00000.png, 00001.png, etc.
    
    if not os.path.exists(mask_path):
        return None, None, None  # No mask available for this frame

    mask = Image.open(mask_path).convert("L")  # Load in grayscale
    mask = mask.resize((W, H))  # Resize to match the rendering resolution
    mask_tensor = T.ToTensor()(mask).to("cuda")  # Convert to tensor [1, H, W]

    # Create normalized coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device="cuda"),  # Normalized Y-coordinates
        torch.linspace(-1, 1, W, device="cuda")   # Normalized X-coordinates
    )

    return mask_tensor, x_grid, y_grid

def render_set( model_path,
                iteration,
                views, 
                gaussians, 
                pipeline, 
                background, 
                interp, 
                extension,
                mask_folder=None,
                save_render_data=False):
    render_path = os.path.join(model_path, f"render")
    render_data_path = os.path.join(model_path, f"render_data") 

    makedirs(render_path, exist_ok=True)
    if save_render_data:
        makedirs(render_data_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        H, W = int(view.image_height), int(view.image_width)
        # Load the mask if a mask folder is provided
        mask_tensor, x_grid, y_grid = (None, None, None)
        if mask_folder:
            mask_data = load_mask(mask_folder, idx, H, W)
            if mask_data[0] is not None:
                mask_tensor, x_grid, y_grid = mask_data


        for i in range(interp):
            result = render(view, gaussians, pipeline, background, interp=interp, interp_idx=i, modify_func=modify_func, mask_tensor=mask_tensor, x_grid=x_grid, y_grid=y_grid)
            rendering = result["render"].cpu()
            # rendering = render(view, gaussians, pipeline, background, interp=interp, interp_idx=i, modify_func=modify_func)["render"].cpu()
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_" + str(i) + extension))

            if save_render_data:
                radii = result["radii"].cpu()
                visibility_filter = result["visibility_filter"].cpu()
                torch.save({
                    "radii": radii,
                    "visibility_filter": visibility_filter
                }, os.path.join(render_data_path, f"{idx:05d}_{i}.pt"))

def render_sets(dataset : ModelParams,
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool, 
                skip_test : bool, 
                interp : int,
                extension: str,
                mask_folder: str,
                save_render_data: bool):
    with torch.no_grad():
        gaussians = gaussianModelRender['gs'](dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, interp, extension, mask_folder=mask_folder, save_render_data=save_render_data)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--poly_degree", type=int, default=1)
    parser.add_argument("--interp", type=int, default=1)
    parser.add_argument("--extension", type=str, default=".png")
    parser.add_argument("--save_render_data", action="store_true")
    parser.add_argument("--mask", type=str, default="NO_MASK", 
                        help="Path to the folder containing per-frame masks")

    args = get_combined_args(parser)
    model.gs_type = "gs"
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.poly_degree = args.poly_degree

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.mask == "NO_MASK":
        args.mask = None

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.interp, args.extension, args.mask, args.save_render_data)