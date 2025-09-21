import torch
import torch.nn as nn
import numpy as np

from sdf import SDF

class SDFLoss(nn.Module):

    def __init__(self, faces, grid_size=32, robustifier=None, debugging=False):
        super(SDFLoss, self).__init__()
        self.sdf = SDF()
        self.register_buffer('faces', torch.tensor(faces, dtype=torch.int32))
        self.grid_size = grid_size
        self.robustifier = robustifier
        self.debugging = debugging

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True

    def filter_isolated_boxes(self, boxes):
        num_people = boxes.shape[0]
        isolated = torch.zeros(num_people, device=boxes.device, dtype=torch.uint8)
        for i in range(num_people):
            isolated_i = False
            for j in range(num_people):
                if j != i:
                    isolated_i |= not self.check_overlap(boxes[i], boxes[j])
            isolated[i] = isolated_i
        return isolated


    def forward(self, human_verts, object_verts, scale_factor=0.2):
        """ Calculate SDF loss between human and object
        remark: currently only supports 1 human and 1 object
        """
        human_verts = human_verts.unsqueeze(0)
        object_verts = object_verts.unsqueeze(0)

        loss = torch.tensor(0., device=object_verts.device)

        box_human = self.get_bounding_boxes(human_verts)
        box_object = self.get_bounding_boxes(object_verts)

        overlapping_boxes = ~self.filter_isolated_boxes(torch.cat([box_human, box_object], dim=0))
        # If no overlapping voxels, return 0
        if overlapping_boxes.sum() == 0:
            return loss

        box_center_human = box_human.mean(dim=1).unsqueeze(dim=1)
        box_scale_human = (1+scale_factor) * 0.5*(box_human[:,1] - box_human[:,0]).max(dim=-1)[0][:,None,None]

        with torch.no_grad():
            human_vertices_centered = human_verts - box_center_human
            human_vertices_centered_scaled = human_vertices_centered / box_scale_human
            assert(human_vertices_centered_scaled.min() >= -1)
            assert(human_vertices_centered_scaled.max() <= 1)
            phi_human = self.sdf(self.faces, human_vertices_centered_scaled)
            assert(phi_human.min() >= 0)

        # Convert vertices to the format expected by grid_sample
        # Change coordinate system to local coordinate system of each object
        vertices_local = (object_verts - box_center_human.unsqueeze(dim=0)) / box_scale_human.unsqueeze(dim=0)
        vertices_grid = vertices_local.view(1,-1,1,1,3)
        # Sample from the phi grid
        phi_val = nn.functional.grid_sample(phi_human[0][None, None], vertices_grid, align_corners=True).view(1, -1)
        cur_loss = phi_val

        if self.debugging:
            import ipdb;ipdb.set_trace()
            
        # robustifier
        if self.robustifier:
            frac = (phi_val / self.robustifier) ** 2
            cur_loss = frac / (frac + 1)

        loss += cur_loss.sum() ** 2
        return loss
