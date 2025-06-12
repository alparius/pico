default_loss_weights = {
    'lw_contact': 8.0,
    'lw_silhouette': 0.3,
    'lw_silhouette_distance': 0.3,
    'lw_scale': 4,
    # 'lw_collision_p2': 100, TODO: collision loss temporarily disabled
    # 'lw_collision_p3': 50, TODO: collision loss temporarily disabled
    'lw_pose_reg': 0.05,
    'lw_silhouette_human': 0.1,
}

class ConfigPack:
    def __init__(
        self,
        # input file names
        human_inference_file: str = 'osx_human.npz',
        human_detection_file: str = 'human_detection.json',
        object_mesh_file: str = 'object.obj',
        object_detection_file: str = 'object_detection.json',
        contact_mapping_file: str = 'corresponding_contacts.json',
        # optimization nr of steps
        nr_phase_1_steps: int = 250,
        skip_phase_1: bool = False,
        nr_phase_2_steps: int = 1500,
        skip_phase_2: bool = False,
        nr_phase_3_steps: int = 1000,
        skip_phase_3: bool = False,
    ):
        self.human_inference_file = human_inference_file
        self.human_detection_file = human_detection_file
        self.object_mesh_file = object_mesh_file
        self.object_detection_file = object_detection_file
        self.contact_mapping_file = contact_mapping_file

        self.nr_phase_1_steps = nr_phase_1_steps
        self.skip_phase_1 = skip_phase_1
        self.nr_phase_2_steps = nr_phase_2_steps
        self.skip_phase_2 = skip_phase_2
        self.nr_phase_3_steps = nr_phase_3_steps
        self.skip_phase_3 = skip_phase_3


default_config = ConfigPack()
