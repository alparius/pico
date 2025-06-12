import cv2
import numpy as np
import trimesh
import json

from src.constants import IMAGE_SIZE, SMPLX_FACES_PATH
from src.utils.structs import HumanParams, ObjectParams


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    h, w = image.shape[:2]
    r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    w = int(r * w)
    h = int(r * h)
    image = cv2.resize(image, (w, h))

    return image


def load_human_params(human_inference_file: str, human_detection_file: str, imgsize: np.ndarray) -> HumanParams:
    human_npz = np.load(human_inference_file, allow_pickle=True)

    # TODO: provide the OSX inference wrapper

    # check if multiple humans are detected
    if len(human_npz['bbox_2']) > 1:
        raise NotImplementedError("Multiple humans detected")
    
    vertices = human_npz['smpl_vertices'][0]
    faces = np.load(SMPLX_FACES_PATH).astype(int)
    human_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # save the centroid offset
    centroid_offset = human_mesh.centroid
    # center the mesh
    human_mesh.apply_translation(-centroid_offset)

    smplx_params = {
        'betas': human_npz['hps_betas'],
        'body_pose': human_npz['hps_body_pose'],
        'global_orient': human_npz['hps_global_orient'],
        'right_hand_pose': human_npz['hps_right_hand_pose'],
        'left_hand_pose': human_npz['hps_left_hand_pose'],
        'jaw_pose': human_npz['hps_jaw_pose'],
        'leye_pose': human_npz['hps_leye_pose'],
        'reye_pose': human_npz['hps_reye_pose'],
        'expression': human_npz['hps_expression']
    }

    with open(human_detection_file, 'r') as f:
        detection = json.load(f)
    mask = np.array(detection['mask']).astype(float)
    # resize to image size
    mask = cv2.resize(mask, (imgsize[1], imgsize[0]))

    human_params = HumanParams(
        vertices = human_mesh.vertices,
        faces = faces,
        centroid_offset = centroid_offset.copy(),
        bbox = human_npz['bbox_2'][0],
        mask = mask,
        smplx_params = smplx_params
    )

    human_params.to_cuda()
    return human_params
    

def load_object_params(object_mesh_file: str, object_detection_file: str, imgsize: np.ndarray) -> ObjectParams:
    obj_mesh = trimesh.load(object_mesh_file)

    # rotate object 90 degrees around x-axis (mostly upright in objaverse)
    obj_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

    # center the mesh
    obj_mesh.apply_translation(-obj_mesh.centroid)

    # load object mask and resize to image size
    with open(object_detection_file, 'r') as f:
        detection = json.load(f)
    mask = np.array(detection['mask']).astype(float)
    mask = cv2.resize(mask, (imgsize[1], imgsize[0]))

    object_params = ObjectParams(
        vertices = obj_mesh.vertices,
        faces = obj_mesh.faces,
        mask = mask,
        scale = obj_mesh.extents.max()
    )

    object_params.to_cuda()
    return object_params

