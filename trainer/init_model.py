from .points_smplx import TrainWrapper as s2a_points
from .smplx_face import TrainWrapper as s2a_face
from .smplx_body_vq import TrainWrapper as s2a_body_vq
from .smplx_body_retnet import TrainWrapper as s2a_body_retnet
import torch

def init_model(model_name, args, config):
    if model_name == 's2a_points':
        generator = s2a_points(
            args,
            config,
        )
    elif model_name == 's2a_face':
        generator = s2a_face(
            args,
            config,
        )
    elif model_name == 's2a_body_vq':
        generator = s2a_body_vq(
            args,
            config,
        )
    elif model_name == 's2a_body_retnet':
        generator = s2a_body_retnet(
            args,
            config,
        )
    else:
        raise ValueError
    return generator


