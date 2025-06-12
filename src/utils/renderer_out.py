import numpy as np
import trimesh
import pyrender
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
)

from src.constants import OSX_FOCAL_VIRTUAL, OSX_INPUT_BODY_SHAPE, OSX_PRINCPT


def get_camera_params(human_bbox):
    # if torch, go back to numpy
    if hasattr(human_bbox, 'detach'):
        human_bbox = human_bbox.detach().cpu().numpy()

    focal = [OSX_FOCAL_VIRTUAL[0] / OSX_INPUT_BODY_SHAPE[1] * human_bbox[2], OSX_FOCAL_VIRTUAL[1] / OSX_INPUT_BODY_SHAPE[0] * human_bbox[3]]
    princpt = [OSX_PRINCPT[0] / OSX_INPUT_BODY_SHAPE[1] * human_bbox[2] + human_bbox[0], OSX_PRINCPT[1] / OSX_INPUT_BODY_SHAPE[0] * human_bbox[3] + human_bbox[1]]
    return focal, princpt

def get_camera_params_torch(human_bbox):
    fv1 = torch.tensor(OSX_FOCAL_VIRTUAL[0], dtype=torch.float32, device='cuda')
    fv2 = torch.tensor(OSX_FOCAL_VIRTUAL[1], dtype=torch.float32, device='cuda')
    bs1 = torch.tensor(OSX_INPUT_BODY_SHAPE[0], dtype=torch.float32, device='cuda')
    bs2 = torch.tensor(OSX_INPUT_BODY_SHAPE[1], dtype=torch.float32, device='cuda')
    pt1 = torch.tensor(OSX_PRINCPT[0], dtype=torch.float32, device='cuda')
    pt2 = torch.tensor(OSX_PRINCPT[1], dtype=torch.float32, device='cuda')

    focal = torch.stack([fv1 / bs2 * human_bbox[2], fv2 / bs1 * human_bbox[3]], dim=0)
    princpt = torch.stack([pt1 / bs2 * human_bbox[2] + human_bbox[0], pt2 / bs1 * human_bbox[3] + human_bbox[1]], dim=0)
    return focal, princpt


def render_overlaid_view(img, mesh, human_bbox, objmesh=None):
    rot = trimesh.transformations.rotation_matrix(
	    np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = get_camera_params(human_bbox)
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img



##############################################
##############################################
##############################################
class MyDifferentiableRenderer:
    def __init__(self, img_shape, faces, human_bbox):
        super().__init__()
        self.img_shape = img_shape
        self.faces = faces
        
        R = torch.tensor(
            [[[ -1.0,  0.0, 0.0],
            [  0.0, -1.0, 0.0],
            [  0.0,  0.0, 1.0]]],
            dtype=torch.float32, device='cuda')
        T = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device='cuda')
        T = torch.bmm(R, T.unsqueeze(-1)).squeeze(-1)

        # Camera parameters
        focal, princpt = get_camera_params_torch(human_bbox) # Adjusted to return tensors

        self.camera = PerspectiveCameras(
            focal_length=focal.unsqueeze(0),
            principal_point=princpt.unsqueeze(0),
            image_size=torch.tensor([[img_shape[0], img_shape[1]]], device='cuda'),
            R=R,
            # T=T,
            in_ndc=False,
            device='cuda'
        )


class MySoftSilhouetteRenderer(MyDifferentiableRenderer):
    def __init__(self, img_shape, faces, human_bbox):
        super().__init__(img_shape, faces, human_bbox)

        blend_params = BlendParams(
            sigma=1e-4,
            gamma=1e-4,
            background_color=[1.0, 1.0, 1.0]
        )
        raster_settings = RasterizationSettings(
            image_size=(img_shape[0], img_shape[1]),
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            bin_size=0,
            # max_faces_per_bin=5000,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def render(self, vertices):
        # # Initialize each vertex to be white in color.
        # verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
        # textures = TexturesVertex(verts_features=verts_rgb.to(device))

        meshes = Meshes(
            verts=[vertices],
            faces=[self.faces],
            # textures=textures
        )
        image = self.renderer(meshes)
        image = image[0, :, :, 3]  # Assuming the last channel is alpha for visibility
        return image
##############################################
##############################################
##############################################


def render_side_views(trimesh_mesh, image):
    h, w, _ = image.shape

    # flip the mesh upside down
    rot = trimesh.transformations.rotation_matrix(
	    np.radians(180), [1, 0, 0])
    trimesh_mesh.apply_transform(rot)

    # Calculate the bounding box of the mesh
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
    bbox = trimesh_mesh.bounds
    center = (bbox[0] + bbox[1]) / 2
    extents = bbox[1] - bbox[0]
    # Calculate the distance to ensure the object is fully visible
    # This might need some adjustment based on your specific mesh and preferences
    distance = max(extents) * 2 # Simple heuristic; adjust as needed

    ##### PYRENDER SETUP
    # Create an orthographic camera
    camera = pyrender.OrthographicCamera(xmag=1.5, ymag=1.5, znear=0.05, zfar=1000.0)
    # Create a directional light source
    # Here, we assume you want the light to shine upwards as well
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    # Use an offscreen renderer to render the scene
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    ##### BACK VIEW
    camera_pose = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, distance],
        [0, 0, 0, 1]
    ])
    camera_pose[:3, 3] = center
    camera_pose[2, 3] += distance
    scene = pyrender.Scene()
    scene.add(mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    back_view, _ = r.render(scene)

    ##### LEFT VIEW
    camera_pose = np.array([
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, distance],
        [0, 0, 0, 1]
    ])
    camera_pose[:3, 3] = center - [distance, 0, 0]
    scene = pyrender.Scene()
    scene.add(mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    left_view, _ = r.render(scene)

    ##### RIGHT VIEW
    camera_pose = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, distance],
        [0, 0, 0, 1]
    ])
    camera_pose[:3, 3] = center + [distance, 0, 0]
    scene = pyrender.Scene()
    scene.add(mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    right_view, _ = r.render(scene)

    ##### TOP-DOWN VIEW
    camera_pose = np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, distance],
        [0, 0, 0, 1]
    ])
    camera_pose[:3, 3] = center
    camera_pose[1, 3] -= distance
    scene = pyrender.Scene()
    scene.add(mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    top_view, _ = r.render(scene)

    ##### FRONT VIEW
    camera_pose = np.array([
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, distance],
        [0, 0, 0, 1]
    ])
    camera_pose[:3, 3] = center - [distance, 0, 0]
    scene = pyrender.Scene()
    scene.add(mesh, pose=np.eye(4))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    front_view, _ = r.render(scene)

    return top_view, left_view, right_view, back_view, front_view
