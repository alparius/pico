import torch

class HumanParams:
    def __init__(
        self,
        vertices,
        faces,
        centroid_offset,
        bbox,
        mask,
        smplx_params,
    ):
        self.vertices = vertices
        self.faces = faces
        self.centroid_offset = centroid_offset
        self.bbox = bbox
        self.mask = mask
        self.smplx_params = smplx_params

    def to_cuda(self):
        self.vertices = torch.from_numpy(self.vertices).float().cuda()
        self.faces = torch.from_numpy(self.faces).float().cuda()
        self.centroid_offset = torch.from_numpy(self.centroid_offset).float().cuda()
        self.bbox = torch.from_numpy(self.bbox).float().cuda()
        self.mask = torch.tensor(self.mask).cuda()
    
    def __str__(self):
        return ('HumanParams:\n'
            f'  - Vertices: {self.vertices.shape}, {type(self.vertices)} {self.vertices[:5]}\n'
            f'  - Faces: {self.faces.shape}, {type(self.faces)} {self.faces[:5]}\n'
            f'  - Centroid offset: {type(self.centroid_offset)} {self.centroid_offset}\n'
            f'  - Bbox: {type(self.bbox)} {self.bbox}\n'
            f'  - Mask: {self.mask.shape}')


class ObjectParams:
    def __init__(
        self,
        vertices,
        faces,
        mask,
        scale,
    ):
        self.vertices = vertices
        self.faces = faces
        self.mask = mask
        self.scale = scale

    def to_cuda(self):
        self.vertices = torch.from_numpy(self.vertices).float().cuda()
        self.faces = torch.from_numpy(self.faces).float().cuda()
        self.mask = torch.tensor(self.mask).cuda()
        self.scale = torch.tensor(self.scale).float().cuda()

    def __str__(self):
        return ('ObjectParams:\n'
            f'  - Vertices: {self.vertices.shape}, {type(self.vertices)} {self.vertices[:5]}\n'
            f'  - Faces: {self.faces.shape}, {type(self.faces)} {self.faces[:5]}\n'
            f'  - Mask: {self.mask.shape}, {type(self.mask)} {self.mask[:5]}\n'
            f'  - Scale: {self.scale}')
