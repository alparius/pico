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

- [2025/06/10] PICO-fit* optimization script is released!

## Installation and Setup
1. First, clone the repo. Then, we recommend creating a clean [conda](https://docs.conda.io/) environment, activating it and installing torch and torchvision, as follows:
```shell
git clone https://github.com/sha2nkt/pico.git
cd pico
conda create -n pico python=3.9 -y
conda activate pico
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
Please adjust the CUDA version as required.

2. Install PyTorch3D from source. Users may also refer to [PyTorch3D-install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details.
However, our tests show that installing using ``conda`` sometimes runs into dependency conflicts.
Hence, users may alternatively install Pytorch3D from source following the steps below.
```shell
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install .
cd ..
```

3. Install the other dependancies and download the required data.
```bash
pip install -r requirements.txt
sh fetch_data.sh
```

4. Please download [SMPL](https://smpl.is.tue.mpg.de/) (version 1.1.0) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) (v1.1) files into the data folder. Please rename the SMPL files to ```SMPL_FEMALE.pkl```, ```SMPL_MALE.pkl``` and ```SMPL_NEUTRAL.pkl```. The directory structure for the ```data``` folder has been elaborated below:

```
├── preprocess
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   ├── SMPL_NEUTRAL.pkl
│   ├── smpl_neutral_geodesic_dist.npy
│   ├── smpl_neutral_tpose.ply
│   ├── smplpix_vertex_colors.npy
├── smplx
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_MALE.pkl
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_NEUTRAL.pkl
│   ├── smplx_neutral_tpose.ply
├── weights
│   ├── pose_hrnet_w32_256x192.pth
├── J_regressor_extra.npy
├── base_dataset.py
├── mixed_dataset.py
├── smpl_partSegmentation_mapping.pkl
├── smpl_vert_segmentation.json
└── smplx_vert_segmentation.json
```
<a name="damon-data-description"></a>
### Download the DAMON dataset

⚠️ Register account on the [PICO website](https://pico.is.tue.mpg.de/register.php), and then use your username and password to login to the _Downloads_ page.

Follow the instructions on the _Downloads_ page to download the DAMON dataset. The provided metadata in the `npz` files is described as follows: 
- `imgname`: relative path to the image file
- `pose` : SMPL pose parameters inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
- `transl` : SMPL root translation inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
- `shape` : SMPL shape parameters inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
- `cam_k` : camera intrinsic matrix inferred from [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
- `polygon_2d_contact`: 2D contact annotation from [HOT](https://hot.is.tue.mpg.de/)
- `contact_label`: 3D contact annotations on the SMPL mesh
- `contact_label_smplx`: 3D contact annotation on the SMPL-X mesh
- `contact_label_objectwise`: 3D contact annotations split into separate object labels on the SMPL mesh
- `contact_label_smplx_objectwise`: 3D contact annotations split into separate object labels on the SMPL-X mesh
- `scene_seg`: path to the scene segmentation map from [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- `part_seg`: path to the body part segmentation map

The order of values is the same for all the keys. 

<a name="convert-damon"></a>
#### Converting DAMON contact labels to SMPL-X format (and back)

To convert contact labels from SMPL to SMPL-X format and vice-versa, run the following command
```bash
python reformat_contacts.py \
    --contact_npz datasets/Release_Datasets/damon/hot_dca_trainval.npz \
    --input_type 'smpl'
```

## Run demo on images
The following command will run PICO on all images in the specified `--img_src`, and save rendering and colored mesh in `--out_dir`. The `--model_path` flag is used to specify the specific checkpoint being used. Additionally, the base mesh color and the color of predicted contact annotation can be specified using the `--mesh_colour` and `--annot_colour` flags respectively. 
```bash
python inference.py \
    --img_src example_images \
    --out_dir demo_out
```

## Training and Evaluation

We release 3 versions of the PICO model:
<ol>
    <li> PICO-HRNet (<em> Best performing model </em>) </li>
    <li> PICO-HRNet w/o context branches </li>
    <li> PICO-Swin </li>
</ol>

All the checkpoints have been downloaded to ```checkpoints```. 
However, please note that versions 2 and 3 have been trained solely on the RICH dataset. <br>
We recommend using the first PICO version.

Please download the actual DAMON dataset from the website and place it in ```datasets/Release_Datasets``` following the instructions given.

### Evaluation
To run evaluation on the DAMON dataset, please run the following command:

```bash
python tester.py --cfg configs/cfg_test.yml
```

### Training
The config provided (```cfg_train.yml```) is set to train and evaluate on all three datasets: DAMON, RICH and PROX. To change this, please change the value of the key ```TRAINING.DATASETS``` and ```VALIDATION.DATASETS``` in the config (please also change ```TRAINING.DATASET_MIX_PDF``` as required). <br>
Also, the best checkpoint is stored by default at ```checkpoints/Other_Checkpoints```.
Please run the following command to start training of the PICO model:

```bash
python train.py --cfg configs/cfg_train.yml
```

### Training on custom datasets

To train on other datasets, please follow these steps:
1. Please create an npz of the dataset, following the structure of the datasets in ```datasets/Release_Datasets``` with the corresponding keys and values.
2. Please create scene segmentation maps, if not available. We have used [Mask2Former](https://github.com/facebookresearch/Mask2Former) in our work.
3. For creating the part segmentation maps, this [sample script](https://github.com/sha2nkt/pico/blob/main/utils/get_part_seg_mask.py) can be referred to.
4. Add the dataset name(s) to ```train.py``` ([these lines](https://github.com/sha2nkt/pico/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/train.py#L83)), ```tester.py``` ([these lines](https://github.com/sha2nkt/pico/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/tester.py#L51)) and ```data/mixed_dataset.py``` ([these lines](https://github.com/sha2nkt/pico/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/data/mixed_dataset.py#L17)), according to the body model being used (SMPL/SMPL-X)
5. Add the path(s) to the dataset npz(s) to ```common/constants.py``` ([these lines](https://github.com/sha2nkt/pico/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/common/constants.py#L19)).
6. Finally, change ```TRAINING.DATASETS``` and ```VALIDATION.DATASETS``` in the config file and you're good to go!

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

### License

See [LICENSE](LICENSE).

### Acknowledgments

We thank Felix Grüninger for advice on mesh preprocessing, Jean-Claude Passy and Valkyrie Felso for advice on the data collection, and Xianghui Xie for advice on HDM evaluation. We also thank Tsvetelina Alexiadis, Taylor Obersat, Claudia Gallatz, Asuka Bertler, Arina Kuznetcova, Suraj Bhor, Tithi Rakshit, Tomasz Niewiadomski, Valerian Fourel and Florentin Doll for their immense help in the data collection and verification process, Benjamin Pellkofer for IT support, and Nikos Athanasiou for the helpful discussions. This work was funded in part by the International Max Planck Research School for Intelligent Systems (IMPRS-IS). D. Tzionas is supported by the ERC Starting Grant (project STRIPES, 101165317).

Dimitris Tzionas has received a research gift fund from Google. While Michael J. Black is a co-founder and Chief Scientist at Meshcapade, his research in this project was performed solely at, and funded solely by, the Max Planck Society.

### Contact

For technical questions, please create an issue. For other questions, please contact `pico@tue.mpg.de`.

For commercial licensing, please contact `ps-licensing@tue.mpg.de`.
