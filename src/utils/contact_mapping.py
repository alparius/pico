import numpy as np
import pickle
import json
import torch

from src.constants import SMPL_TO_SMPLX_MATRIX_PATH
from src.utils.geometry import rot6d_to_matrix


def load_contact_mapping(contact_map_path: str, convert_to_smplx: bool = True) -> dict:
    contact_mapping_json = json.load(open(contact_map_path))

    # simplify the dictionary
    contact_transfer_map = {}
    for elem in contact_mapping_json['data']:
        contact_transfer_map[elem['name']] = elem['contactPoints']

    # remove unmapped contact patches
    # human keys start with 'human', object keys start with 'object'
    # remove any keys, that don't have a matching other key
    keys_to_remove = []
    for key in contact_transfer_map:
        if key.startswith('human'):
            if not any([x == key.replace('human', 'obj') for x in contact_transfer_map]):
                keys_to_remove.append(key)
        elif key.startswith('obj'):
            if not any([x == key.replace('obj', 'human') for x in contact_transfer_map]):
                keys_to_remove.append(key)
    for key in keys_to_remove:
        contact_transfer_map.pop(key)
        print(f"removed {key} from contact_transfer_map")

    # check if the same number of keys starting with 'human' and 'object' are present
    human_keys = [x for x in contact_transfer_map if x.startswith('human')]
    object_keys = [x for x in contact_transfer_map if x.startswith('obj')]
    assert len(human_keys) == len(object_keys), f"number of human keys: {len(human_keys)}, number of object keys: {len(object_keys)}"

    # convert the human part to smplx if needed
    if convert_to_smplx:
        contact_transfer_map = convert_contact_map_to_smplx(contact_transfer_map)

    # print the number of contact points for each object
    print('contact_transfer_map processed:')
    for key in contact_transfer_map:
        if key.startswith('human'):
            print(f"... {key.replace('human', '')}: h {len(contact_transfer_map[key])}, o {len(contact_transfer_map[key.replace('human', 'obj')])}")
            assert len(contact_transfer_map[key]) == len(contact_transfer_map[key.replace('human', 'obj')]), f"number of contact points for {key} and {key.replace('human', 'obj')} don't match"

    return contact_transfer_map


def convert_contact_map_to_smplx(contact_transfer_map: dict) -> dict:
    # load smplx to smpl matrix
    with open(SMPL_TO_SMPLX_MATRIX_PATH, 'rb') as f:
        smpl_to_smplx = pickle.load(f)

    for key in contact_transfer_map:
        if key.startswith('human'):
            smplx_contact = []
            for v in contact_transfer_map[key]:
                v = int(v[1:])
                smpl_vertices = np.zeros((6890))
                smpl_vertices[v] = 1
                smplx_vertices = smpl_vertices @ smpl_to_smplx['matrix'].T
                    
                best = np.argmax(smplx_vertices)
                smplx_contact.append('v ' + str(best))

            contact_transfer_map[key] = smplx_contact

    return contact_transfer_map


def interpret_contact_points(contact_map, human_vertices, obj_mesh):
    human_points = []
    object_points = []
    for shapekey in contact_map:
        if shapekey.startswith('humanShape'):  # Processing human mesh points
            for point in contact_map[shapekey]:
                if point.startswith('v'):  # Single vertex
                    idx = int(point.split()[1])
                    human_points.append(human_vertices[idx])
        elif shapekey.startswith('objShape'):  # Processing object mesh points
            for point in contact_map[shapekey]:
                if point.startswith('f'):  # Face with Barycentric coordinates
                    parts = point.split()
                    face_idx = int(parts[1])
                    bary_coords = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    # Find the vertices of the face
                    vertices = obj_mesh.vertices[obj_mesh.faces[face_idx]]
                    # Calculate the point using Barycentric coordinates
                    point = vertices[0] * bary_coords[0] + vertices[1] * bary_coords[1] + vertices[2] * (1 - bary_coords[0] - bary_coords[1])
                    object_points.append(point)
                elif point.startswith('e'):  # Edge
                    raise NotImplementedError("Edge contact points are not supported yet")
                elif point.startswith('v'):
                    raise NotImplementedError("Single vertex contact points are not supported yet")

    human_points = torch.from_numpy(np.asarray(human_points)).float().cuda()
    object_points = torch.from_numpy(np.asarray(object_points)).float().cuda()
    return human_points, object_points


def calculate_human_points(transformed_vertices, contact_data):
    human_points = []
    for shapekey in contact_data:
        if shapekey.startswith('humanShape'):  # Processing human mesh points
            for point in contact_data[shapekey]:
                if point.startswith('v'):  # Single vertex
                    idx = int(point.split()[1])
                    human_points.append(transformed_vertices[idx])
    # Stack the list of tensors into a single tensor
    human_points_tensor = torch.stack(human_points)
    return human_points_tensor


def calculate_object_points(transformed_vertices, contact_data, mesh_obj_faces):
    object_points = []
    for shapekey in contact_data:
        if shapekey.startswith('objShape'):  # Processing object mesh points
            for point in contact_data[shapekey]:
                if point.startswith('f'):  # Face with Barycentric coordinates
                    parts = point.split()
                    face_idx = int(parts[1])
                    bary_coords = torch.tensor([float(parts[2]), float(parts[3]), float(parts[4])], dtype=torch.float32)
                    
                    # Find the vertices of the face using PyTorch indexing
                    try:
                        fvi = mesh_obj_faces[face_idx]
                        face_vertex_indices = fvi.clone().detach().to(torch.long)
                        vertices = transformed_vertices[face_vertex_indices]
                    except:
                        print(f"face_idx: {face_idx}, len(mesh_obj_faces): {mesh_obj_faces.shape}")
                        
                    
                    # Calculate the point using Barycentric coordinates with PyTorch operations
                    point = vertices[0] * bary_coords[0] + vertices[1] * bary_coords[1] + vertices[2] * (1 - bary_coords[0] - bary_coords[1])
                    object_points.append(point)

    # Stack the list of tensors into a single tensor
    object_points_tensor = torch.stack(object_points)
    
    return object_points_tensor


def apply_transformation(vertices, rot, translation, scaling=1.0, rotmat=False):
    """Apply transformation using 6D rotation representation.
    inputs:
    - vertices: tensor of shape (N, 3) containing the vertices of the mesh
    - rot6d: tensor of shape (6,) containing the 6D rotation representation
    - translation: tensor of shape (3,) containing the translation values
    - scaling: scalar tensor containing the scaling factor
    returns:
    - transformed_vertices: tensor of shape (N, 3) containing the transformed vertices
    """
    if not rotmat:
        rot_matrix = rot6d_to_matrix(rot).view(1, 3, 3)
    else:
        rot_matrix = rot
    scaled_vertices = vertices * scaling
    rotated_vertices = torch.matmul(scaled_vertices.unsqueeze(1), rot_matrix).squeeze(1)
    transformed_vertices = rotated_vertices + translation
    
    return transformed_vertices 


def select_pose_parameters(contact_mapping: dict):
    # get body parts that are in contact
    contact_bodyparts = []
    for k, _ in contact_mapping.items():
        if 'humanShape' in k:
            contact_bodyparts.append(k.replace('humanShape', ''))

    # select the joints to optimize
    joint_ids = []
    left_hand_opt = False
    right_hand_opt = False

    if 'lefthand' in contact_bodyparts or 'lefthandback' in contact_bodyparts:
        left_hand_opt = True
        joint_ids.extend([20, 18, 16])
    elif 'leftforearm' in contact_bodyparts:
        joint_ids.extend([18, 16])
    elif 'leftarm' in contact_bodyparts or 'leftupperarm' in contact_bodyparts or 'leftshoulder' in contact_bodyparts:
        joint_ids.extend([16])

    if 'righthand' in contact_bodyparts or 'righthandback' in contact_bodyparts:
        right_hand_opt = True
        joint_ids.extend([21, 19, 17])
    elif 'rightforearm' in contact_bodyparts:
        joint_ids.extend([19, 17])
    elif 'rightarm' in contact_bodyparts or 'rightupperarm' in contact_bodyparts or 'rightshoulder' in contact_bodyparts:
        joint_ids.extend([17])

    if 'leftfoot' in contact_bodyparts or 'leftfoottop' in contact_bodyparts or 'leftankle' in contact_bodyparts:
        joint_ids.extend([10, 7, 4, 1])
    elif 'leftleg' in contact_bodyparts or 'leftshinback' in contact_bodyparts:
        joint_ids.extend([4, 1])
    elif 'leftupleg' in contact_bodyparts or 'leftthighback' in contact_bodyparts or 'leftthighfront' in contact_bodyparts:
        joint_ids.extend([1])

    if 'rightfoot' in contact_bodyparts or 'rightfoottop' in contact_bodyparts or 'rightankle' in contact_bodyparts:
        joint_ids.extend([11, 8, 5, 2])
    elif 'rightleg' in contact_bodyparts or 'rightshinback' in contact_bodyparts:
        joint_ids.extend([5, 2])
    elif 'rightupleg' in contact_bodyparts or 'rightthighback' in contact_bodyparts or 'rightthighfront' in contact_bodyparts:
        joint_ids.extend([2])

    # remove duplicates
    joint_ids = list(set(joint_ids))
    # convert joint ids to smplx body pose indices
    body_pose_indices_to_opt = []
    for j in joint_ids:
        body_pose_indices_to_opt.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])

    return body_pose_indices_to_opt, left_hand_opt, right_hand_opt
