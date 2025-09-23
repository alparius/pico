# PICO: Reconstructing 3D People In Contact with Objects [CVPR 2025]

> Code repository for the paper:  
> [**PICO: Reconstructing 3D People In Contact with Objects**](https://arxiv.org/abs/2504.17695)  
> [Alpár Cseke\*](https://is.mpg.de/person/acseke), [Shashank Tripathi\*](https://sha2nkt.github.io/), [Sai Kumar Dwivedi](https://saidwivedi.in/), [Arjun Lakshmipathy](https://www.andrew.cmu.edu/user/aslakshm/), [Agniv Chatterjee](https://ac5113.github.io/), [Michael J. Black](https://ps.is.mpg.de/person/black), [Dimitrios Tzionas](https://ps.is.mpg.de/person/dtzionas)<br />
> *Conference on Computer Vision and Pattern Recognition (CVPR), 2025* <br />
> \* equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2309.15273-00ff00.svg)](https://arxiv.org/abs/2504.17695)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://pico.is.tue.mpg.de/) [![explore pico-db](https://img.shields.io/badge/explore%20pico--db-up-6c9b1c?style=flat&logo=google-chrome&logoColor=white)](https://pico.is.tue.mpg.de/dataexploration.html)

![teaser](assets/teaser.png)

[[Project Page](https://pico.is.tue.mpg.de)] [[Paper](https://arxiv.org/abs/2504.17695)] [[Video]()] [[Poster](https://pico.is.tue.mpg.de/media/upload/static/images/CVPR2025_PICO_Poster.pdf)] [[License](https://pico.is.tue.mpg.de/license.html)] [[Contact](mailto:pico@tue.mpg.de)]

## News :triangular_flag_on_post:

- [2025/06/11] PICO-fit* optimization script is released!
- [2025/09/10] Added auxiliary files for PICO-fit* optimization as reference
- [2025/09/21] Added back collision loss module, with installation help
- [2025/09/22] Closest match lookup script in PICO-db for new input images
- [2025/09/23] Example script on how to load PICO-db contact mappings

## Installation and Setup
1. First, clone the repo. Then, we recommend creating a clean [conda](https://docs.conda.io/) environment, as follows:
```shell
git clone https://github.com/alparius/pico.git
cd pico
conda create -n pico python=3.10 -y
conda activate pico
```

2. Install packages:
```shell
pip install -r requirements.txt
```

3. Install PyTorch:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Please adjust the CUDA version as required.

4. Install PyTorch3D from source. Users may also refer to [PyTorch3D-install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details.
However, our tests show that installing using ``conda`` sometimes runs into dependency conflicts.
Hence, users may alternatively install Pytorch3D from source.
```shell
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

5. Install the SDF-based collision loss library:
- based on https://github.com/JiangWenPL/multiperson/tree/master/sdf
- go to `src/utils/sdf` and run `python setup.py install`

6. Download some required files:
- run `sh fetch_static.sh` (see the script for details)
- download the smplx model files from [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip). Put `SMPLX_NEUTRAL.npz` under `static/human_model_files/smplx/`

## Download the PICO-db dataset

Register an account on the [PICO website](https://pico.is.tue.mpg.de) to be able to access the subpage to download the dataset. The dataset consists of the selected object mesh for each image and a contact map between the SMPL-X human mesh and the aforementioned object mesh.

## Run the PICO-fit demo

```
python demo.py <folder_path_with_inputs> <folder_path_for_outputs>
```
e.g.:
```
python demo.py demo_input/skateboard__vcoco_000000012938 demo_output/skateboard__vcoco_000000012938
```

The input folder has to include the following files:
- the input image that has the same filename as the folder itself (plus an image extension)
- `osx_human.npz`: human pose and shape data
- `human_detection.npz`, `object_detection.npz`: mask and bbox for the two subjects
- `object.obj`: trimesh file of the object the human interacts with
- `corresponding_contacts.json`: contact mapping data

#### For PICO-db images:
- the latter two files make up the dataset itself that you can download from the above link
- there we also include the other 3 files for most of the samples in another archive, but feel free to bring your own inference results.

#### For brand new images:
-  please refer to the `notebooks/contact_lookup_on_dataset.ipynb` script as an example for doing closest match lookup in PICO-db. This finds the closest contact sample (and corresponding object mesh) in the database given the human contact data, which then can be used to reconstruct the interaction from the new image. See the second cell of the notebook for more details.
- the other 3 `.npz` files you will have to provide yourself with the off-the-shelf methods of your choice


## OpenShape-based object retrieval

Please refer to the following repository for efficient object lookup and retrieval from a single image.  
The same object retrieval strategy was used in both PICO and [InteractVLM](https://interactvlm.is.tue.mpg.de/).

[![GitHub Repo](https://img.shields.io/badge/GitHub-Object__Retrieval-blue?logo=github)](https://github.com/saidwivedi/Object_Retrieval)



## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{cseke_tripathi_2025_pico,
    title     = {{PICO}: Reconstructing {3D} People In Contact with Objects},
    author    = {Cseke, Alp\'{a}r and Tripathi, Shashank and Dwivedi, Sai Kumar and Lakshmipathy, Arjun and Chatterjee, Agniv and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```

## License

See [LICENSE](LICENSE).

## Acknowledgments

We thank Felix Grüninger for advice on mesh preprocessing, Jean-Claude Passy and Valkyrie Felso for advice on the data collection, and Xianghui Xie for advice on HDM evaluation. We also thank Tsvetelina Alexiadis, Taylor Obersat, Claudia Gallatz, Asuka Bertler, Arina Kuznetcova, Suraj Bhor, Tithi Rakshit, Tomasz Niewiadomski, Valerian Fourel and Florentin Doll for their immense help in the data collection and verification process, Benjamin Pellkofer for IT support, and Nikos Athanasiou for the helpful discussions. This work was funded in part by the International Max Planck Research School for Intelligent Systems (IMPRS-IS). D. Tzionas is supported by the ERC Starting Grant (project STRIPES, 101165317).

Dimitris Tzionas has received a research gift fund from Google. While Michael J. Black is a co-founder and Chief Scientist at Meshcapade, his research in this project was performed solely at, and funded solely by, the Max Planck Society.

## Contact

For technical questions, please create an issue. For other questions, please contact `pico@tue.mpg.de`.

For commercial licensing, please contact `ps-licensing@tue.mpg.de`.
