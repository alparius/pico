import os
import numpy as np
import trimesh
from PIL import Image

from src.constants import COLOR_HUMAN_BLUE, COLOR_OBJECT_RED
from src.utils.structs import HumanParams, ObjectParams
from src.utils.renderer_out import render_overlaid_view, render_side_views


def save_phase_results(
    img_filename: str,
    output_folder: str,
    img: np.ndarray,
    human_params: HumanParams,
    object_params: ObjectParams,
    phase: int,
):
    # common folder, already exists if not first image
    os.makedirs(output_folder, exist_ok=True)
    
    # save the combined mesh
    mesh_h = trimesh.Trimesh(vertices=human_params.vertices.detach().cpu().numpy(), faces=human_params.faces.detach().cpu().numpy())
    mesh_h.visual.face_colors = COLOR_HUMAN_BLUE
    mesh_o = trimesh.Trimesh(vertices=object_params.vertices.detach().cpu().numpy(), faces=object_params.faces.detach().cpu().numpy())
    mesh_o.visual.face_colors = COLOR_OBJECT_RED
    mesh = mesh_h + mesh_o
    mesh.export(os.path.join(output_folder, f'{img_filename[:-4]}_phase{phase}.obj'))

    # save rendered views
    if phase == 3:
        visualize_human_object_results(img, img_filename, mesh, human_params, output_folder)

    return


def visualize_human_object_results(img, img_filename, mesh, human_params, output_folder):
    ##### save the images
    frontal = visualize_frontal_overlaid(img, mesh, human_params.centroid_offset, human_params.bbox)
    top_down, left_image, right_image, back_image, front2 = render_side_views(mesh, img)
    
    # compose all 6 images into one
    top_row = np.concatenate((img, frontal, top_down), axis=1)
    bottomrow = np.concatenate((left_image, right_image, back_image), axis=1)
    combined = np.concatenate((top_row, bottomrow), axis=0)
    combined = Image.fromarray(combined)
    combined.save(os.path.join(output_folder, img_filename))


def visualize_frontal_overlaid(img, mesh, human_offset, human_bbox):
    # bring mesh to camera frame
    mesh.apply_translation(human_offset.detach().cpu().numpy())

    # render overlaid view
    vis_img = render_overlaid_view(img, mesh, human_bbox.detach().cpu().numpy())
    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
    return vis_img
