#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel


optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
}

gaussianModel = {
    "gs": GaussianModel,
}

gaussianModelRender = {
    "gs": GaussianModel,
    "pgs": PointsGaussianModel
}
