IMAGE_SIZE = 640

# path to smplx files
SMPLX_FACES_PATH = 'static/smplx_faces.npy'
SMPL_TO_SMPLX_MATRIX_PATH = 'static/smpl_to_smplx.pkl'
# SMPL_TO_SMPLX_MATRIX_PATH = '/ps/scratch/ps_shared/stripathi/deco++/alpar/essentials/models_utils/smpl_to_smplx.pkl'
HUMAN_MODEL_PATH = 'static/human_model_files'

# mesh colors
COLOR_HUMAN_BLUE = [67, 135, 240, 255]
COLOR_OBJECT_RED = [255, 69, 0, 255]

# OSX camera setup
OSX_FOCAL_VIRTUAL = (5000, 5000)
OSX_INPUT_BODY_SHAPE = (256, 192)
OSX_PRINCPT = (OSX_INPUT_BODY_SHAPE[1] / 2, OSX_INPUT_BODY_SHAPE[0] / 2)

# boilerplate
SMPLX_LAYER_ARGS = {
    'create_global_orient': False,
    'create_body_pose': False,
    'create_left_hand_pose': False,
    'create_right_hand_pose': False,
    'create_jaw_pose': False,
    'create_leye_pose': False,
    'create_reye_pose': False,
    'create_betas': False,
    'create_expression': False,
    'create_transl': False
}