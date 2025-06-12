import sys
import os

from src.utils.load_utils import load_image, load_human_params, load_object_params
from src.utils.contact_mapping import load_contact_mapping
from src.utils.save_results import save_phase_results
from src.config_packs import default_config, default_loss_weights
from src.phase1_contact import optimize_phase1_contact
from src.phase2_image import optimize_phase2_image
from src.phase3_human import optimize_phase3_human


def main(
        input_folder: str,
        output_folder: str,
        cfg = None,
        loss_weights = None
    ):
    if cfg is None:
        cfg = default_config
    if loss_weights is None:
        loss_weights = default_loss_weights

    img_filename = input_folder.split("/")[-1]  
    img = load_image(os.path.join(input_folder, img_filename))

    if os.path.exists(os.path.join(output_folder, img_filename)):
        print(f"--> Skipping {img_filename} as it has already been processed.")
        return


    human_params = load_human_params(
        os.path.join(input_folder, cfg.human_inference_file),
        os.path.join(input_folder, cfg.human_detection_file),
        img.shape[:2]
    )
    object_params = load_object_params(
        os.path.join(input_folder, cfg.object_mesh_file),
        os.path.join(input_folder, cfg.object_detection_file),
        img.shape[:2]
    )
    contact_mapping = load_contact_mapping(
        os.path.join(input_folder, cfg.contact_mapping_file)
    )

    
    if not cfg.skip_phase_1:
        p1_object_params = optimize_phase1_contact(human_params, object_params, contact_mapping, cfg.nr_phase_1_steps)
        object_params.vertices = p1_object_params['vertices']
        save_phase_results(
            img_filename, output_folder, img,
            human_params, object_params,
            phase = 1,
        )


    if not cfg.skip_phase_2:
        p2_object_params = optimize_phase2_image(human_params, object_params, contact_mapping, img, loss_weights, cfg.nr_phase_2_steps)
        object_params.vertices = p2_object_params['vertices']
        object_params.scale = p2_object_params['scaling']
        save_phase_results(
            img_filename, output_folder, img,
            human_params, object_params,
            phase = 2,
        )


    if not cfg.skip_phase_3:
        p3_human_params = optimize_phase3_human(human_params, object_params, contact_mapping, img, loss_weights, cfg.nr_phase_3_steps)
        human_params.vertices = p3_human_params['vertices']
        save_phase_results(
            img_filename, output_folder, img,
            human_params, object_params,
            phase = 3,
        )



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python demo.py <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.exists(input_folder):
        print("---> Folder does not exist: ", input_folder)
        sys.exit(1)

    main(input_folder, output_folder)
