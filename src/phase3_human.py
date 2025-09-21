import torch
import torch.nn as nn
import trimesh
from tqdm.auto import tqdm
import numpy as np
import smplx

from src.constants import HUMAN_MODEL_PATH, SMPLX_LAYER_ARGS
from src.utils.contact_mapping import calculate_human_points, interpret_contact_points, select_pose_parameters
from src.utils.renderer_out import MySoftSilhouetteRenderer
from src.utils.structs import HumanParams, ObjectParams
from src.utils.sdf.sdf.sdf_loss import SDFLoss


class Phase_3_Optimizer(nn.Module):
    def __init__(self,
        smplx_params,
        human_points,
        object_points,
        contact_mapping,
        body_pose_indices_to_opt,
        left_hand_opt,
        right_hand_opt,
        human_params,
        object_params,
        img
    ):
        super(Phase_3_Optimizer, self).__init__()

        self.register_buffer('smplx_betas', torch.tensor(smplx_params['betas']).float().cuda())
        self.register_buffer('smplx_global_orient', torch.tensor(smplx_params['global_orient']).float().cuda())
        self.register_buffer('smplx_jaw_pose', torch.tensor(smplx_params['jaw_pose']).float().cuda())
        self.register_buffer('smplx_leye_pose', torch.tensor(smplx_params['leye_pose']).float().cuda())
        self.register_buffer('smplx_reye_pose', torch.tensor(smplx_params['reye_pose']).float().cuda())
        self.register_buffer('smplx_expression', torch.tensor(smplx_params['expression']).float().cuda())

        self.body_pose_indices_to_opt = body_pose_indices_to_opt
        self.left_hand_opt = left_hand_opt
        self.right_hand_opt = right_hand_opt

        self.smplx_body_pose_opt = nn.Parameter(
            torch.tensor(smplx_params['body_pose'][:, self.body_pose_indices_to_opt]).float().cuda(), 
            requires_grad=True)
        self.register_buffer('smplx_body_pose_init', torch.tensor(smplx_params['body_pose']).float().cuda())

        if left_hand_opt:
            self.smplx_left_hand_pose = nn.Parameter(torch.tensor(smplx_params['left_hand_pose']).float().cuda(), requires_grad=True)
        else:
            self.register_buffer('smplx_left_hand_pose', torch.tensor(smplx_params['left_hand_pose']).float().cuda())
        self.register_buffer('smplx_left_hand_pose_init', torch.tensor(smplx_params['left_hand_pose']).float().cuda())
        if right_hand_opt:
            self.smplx_right_hand_pose = nn.Parameter(torch.tensor(smplx_params['right_hand_pose']).float().cuda(), requires_grad=True)
        else:
            self.register_buffer('smplx_right_hand_pose', torch.tensor(smplx_params['right_hand_pose']).float().cuda())
        self.register_buffer('smplx_right_hand_pose_init', torch.tensor(smplx_params['right_hand_pose']).float().cuda())

        self.register_buffer('human_points', human_points)
        self.register_buffer('object_points', object_points)
        self.contact_transfer_map = contact_mapping

        smplx_model = smplx.create(HUMAN_MODEL_PATH, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **SMPLX_LAYER_ARGS)
        self.smplx_model = smplx_model.cuda()

        self.register_buffer('hum_vertices', human_params.vertices)
        self.register_buffer('hum_faces', human_params.faces)
        self.register_buffer('hum_centroid_offset', human_params.centroid_offset)
        self.register_buffer('hum_bbox', human_params.bbox)
        self.register_buffer('hum_mask', human_params.mask.float())
        self.register_buffer('obj_vertices', object_params.vertices)
        self.img = img

        # smplx -> OSX vertex offset
        newverts = self.get_human_verts(remove_offset=False)
        temp_mesh = trimesh.Trimesh(vertices=newverts.detach().cpu().numpy(), faces=human_params.faces.detach().cpu().numpy())
        self.smplx_offset = torch.tensor(temp_mesh.centroid).float().cuda()

        self.renderer = MySoftSilhouetteRenderer(img.shape, human_params.faces, human_params.bbox)

        # SDF collision loss setup
        self.sdf_loss = SDFLoss(human_params.faces, robustifier=1.0)


    def get_smplx_body_pose(self):
        # Recombine the optimized and constant parameters
        full_pose = self.smplx_body_pose_init.clone()
        full_pose[:, self.body_pose_indices_to_opt] = self.smplx_body_pose_opt
        return full_pose

    def get_human_verts(self, remove_offset=True):
        output = self.smplx_model(
            betas=self.smplx_betas,
            body_pose=self.get_smplx_body_pose(),
            global_orient=self.smplx_global_orient,
            right_hand_pose=self.smplx_right_hand_pose,
            left_hand_pose=self.smplx_left_hand_pose,
            jaw_pose=self.smplx_jaw_pose,
            leye_pose=self.smplx_leye_pose,
            reye_pose=self.smplx_reye_pose,
            expression=self.smplx_expression
        )
        verts_person = output.vertices[0]
        if remove_offset:
            verts_person = verts_person - self.smplx_offset
        return verts_person


    def calculate_contact_loss(self, upd_human_vertices):
        new_human_points = calculate_human_points(upd_human_vertices, self.contact_transfer_map)
        loss = torch.nn.functional.mse_loss(new_human_points, self.object_points)
        return {"loss_contact": loss}

    def calculate_collision_loss(self, upd_human_vertices):
        loss = self.sdf_loss(upd_human_vertices, self.obj_vertices)
        return {"loss_collision_p3": loss}
    
    def calculate_pose_reg_loss(self):
        loss = torch.nn.functional.mse_loss(self.smplx_body_pose_opt, self.smplx_body_pose_init[:, self.body_pose_indices_to_opt])
        if self.left_hand_opt:
            loss += torch.nn.functional.mse_loss(self.smplx_left_hand_pose, self.smplx_left_hand_pose_init)
        if self.right_hand_opt:
            loss += torch.nn.functional.mse_loss(self.smplx_right_hand_pose, self.smplx_right_hand_pose_init)
        return {"loss_pose_reg": loss}
    

    def calculate_silhouette_loss_iou(self, upd_human_vertices):
        current_mask = self.renderer.render(
            upd_human_vertices + self.hum_centroid_offset
        )
        intersection = torch.sum(current_mask * self.hum_mask)
        union = torch.sum((current_mask + self.hum_mask).clamp(0, 1))
        loss = 1 - intersection / union
        return {"loss_silhouette_human": loss}

    def calculate_silhouette_loss_l2(self, upd_human_vertices):
        loss = torch.tensor(0.0).cuda()
        pred_mask = self.renderer.render(upd_human_vertices + self.hum_centroid_offset)
        loss = torch.nn.functional.mse_loss(pred_mask, self.hum_mask)
        return {"loss_silhouette_human": loss}


    def forward(self, loss_weights: dict):
        upd_human_vertices = self.get_human_verts()

        loss_dict = {}
        if loss_weights["lw_contact"] > 0:
            loss_dict.update(self.calculate_contact_loss(upd_human_vertices))
        if loss_weights["lw_collision_p3"] > 0:
            loss_dict.update(self.calculate_collision_loss(upd_human_vertices))
        if loss_weights["lw_pose_reg"] > 0:
            loss_dict.update(self.calculate_pose_reg_loss())
        if loss_weights["lw_silhouette_human"] > 0:
            loss_dict.update(self.calculate_silhouette_loss_iou(upd_human_vertices))

        return loss_dict
   

def optimize_phase3_human(
    human_params: HumanParams, object_params: ObjectParams, contact_mapping: dict, img: np.ndarray, loss_weights: dict, nr_phase_3_steps: int,
):
    object_mesh = trimesh.Trimesh(vertices=object_params.vertices.detach().cpu().numpy(), faces=object_params.faces.detach().cpu().numpy())
    human_mesh = trimesh.Trimesh(vertices=human_params.vertices.detach().cpu().numpy(), faces=human_params.faces.detach().cpu().numpy())

    human_points, object_points = interpret_contact_points(contact_mapping, human_mesh.vertices, object_mesh)

    # select which pose parameters (and hands) to optimize - the ones in contact
    body_pose_indices_to_opt, left_hand_opt, right_hand_opt = select_pose_parameters(contact_mapping)
    if len(body_pose_indices_to_opt) == 0 and not left_hand_opt and not right_hand_opt:
        print("--> No contacting limbs to optimize! Skipping phase 3.")
        human_parameters = {}
        human_parameters["vertices"] = human_params.vertices.detach()
        return human_parameters, {}
    print("Optimizing body pose indices:", body_pose_indices_to_opt)
    print("Optimizing left hand:", left_hand_opt)
    print("Optimizing right hand:", right_hand_opt)

    model = Phase_3_Optimizer(
        human_params.smplx_params,
        human_points,
        object_points,
        contact_mapping,
        body_pose_indices_to_opt,
        left_hand_opt,
        right_hand_opt,
        human_params,
        object_params,
        img,
    )
    model.cuda()

    # optimizer params
    opt_params = [
        {'params': [model.smplx_body_pose_opt], 'lr': 0.01},
    ]
    if left_hand_opt:
        opt_params.append({'params': [model.smplx_left_hand_pose], 'lr': 0.01})
    if right_hand_opt:
        opt_params.append({'params': [model.smplx_right_hand_pose], 'lr': 0.01})

    # optimizer with separate learning rates for each parameter
    optimizer = torch.optim.Adam(opt_params)

    loop = tqdm(total=nr_phase_3_steps)
    for i in range(nr_phase_3_steps):
        optimizer.zero_grad()
        loss_dict = model(loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        loss.backward(retain_graph=True)
        optimizer.step()
        loop.set_description(f'loss: {loss.item():.3g}')
        loop.update()

        if i % 10 == 0:
            loss_str = " | ".join([f"{k}: {loss_dict_weighted[k].item():.3g}" for k in loss_dict_weighted])
            print(loss_str)
            print('body', torch.mean(model.smplx_body_pose_opt.grad), torch.mean(model.smplx_body_pose_opt))
            if left_hand_opt:
                print('lhand', torch.mean(model.smplx_left_hand_pose.grad), torch.mean(model.smplx_left_hand_pose))
            if right_hand_opt:
                print('rhand', torch.mean(model.smplx_right_hand_pose.grad), torch.mean(model.smplx_right_hand_pose))


    human_parameters = {}
    updated_human_vertices = model.get_human_verts()
    human_parameters["vertices"] = updated_human_vertices.detach()

    return human_parameters
