import torch
from torch.nn import functional as F
import numpy as np


def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        rot_6d (B x 6): Batch of 6D Rotation representation.
    Returns:
        Rotation matrices (B x 3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def largest_extent(point_cloud):
    # Ensure point_cloud is a numpy array
    point_cloud = np.array(point_cloud)
    # Find the minimum and maximum coordinates along each axis
    min_coords = np.amin(point_cloud, axis=0)
    max_coords = np.amax(point_cloud, axis=0)
    # Calculate the extents along each axis
    extents = max_coords - min_coords
    # Find the largest extent
    largest_extent = np.amax(extents)
    return largest_extent
