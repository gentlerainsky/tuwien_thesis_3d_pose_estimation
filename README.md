# 3D Pose Estimation

This repository is a part of the Synthetic Cabin Project

## Project Structure

- `modules/`
  - `data_preprocessing/` - contains data preprocessor for each dataset. The main jobs of the proprocessor are to format input images and their annotation into an COCO-style dataset.
  - `human_detector/` and `pose_estimator_2d/`
    - `config/` - contains MMPose / MMDet model configuration.
    - `human_detector.py` and `pose_estimator_2d.py` - contains logic for model finetuning, and inference.
    - `main.py` - a helper function to run the models over all images.
  - `lifter_2d_3d/`
    - `dataset/`
      - `base_dataset.py` - used for Synthetic Cabin IR
      - `drive_and_act_keypoint_dataset.py` - used for Drive&Act Dataset
      - `gan_keypoint_dataset.py` - used specifically for the RepNet model.
      - `synthetic_cabin_ir_1m_dataset.py` - used for Synthetic Cabin IR 1M
    - `utils/`
    - `model/`
      - `common/` - Common Pytorch Lightning model implementation for all of the models.
      - `linear_model/` - From [Martinez et al. (2017)](https://arxiv.org/abs/1705.03098). Implementation from [motokimura's GitHub](https://github.com/motokimura/3d-pose-baseline-pytorch) is used.
      - `semgcn/` - Implementation from [Zhao et al. (2019)](https://github.com/garyzhao/SemGCN)
      - `graph_mlp/` - Implementation from [Li et al. (2022)](https://github.com/Vegetebird/GraphMLP)
      - `graformer/` - Implementation from [Zhao et al. (2022)](https://github.com/Graformer/GraFormer)
      - `jointformer/` - Implementation from [Lutz et al. (2022)](https://github.com/seblutz/JointFormer)
      - `repnet/` - Implement PyTorch version based on TensorFlow implementation from [Bastian Wandt's GitHub](https://github.com/bastianwandt/RepNet).
  - `experiments/` - Contain functions to help running experiemnt including common dataset loaders, result summarizer, and trainer.
  - `utils/` - helper functions such as visualization.
- `experiments/` - Contain script for runing experiment, and analysis of the results.
- `demo/` - Contain data exploration notebooks, and some example model inference output.
- `script/` - Contain data preprocessing scripts.

## Installation

- Packages used are listed in `Dockerfile`.
- To use Docker images, we can run the command in `Makefile`.

```bash
# To build the image.
make build
# To run the shell of the image. Don't forget to change the `-v` option to suit the running environment.
make shell
# To run jupyter server
make jupyter
# or
make jupyter-bg
```

## Data Preprocessor

The main task of data preprocessors is to extract images from the original folder or from videos (for Drive&Act), process annotation, and save them into to the COCO-style data folder. All of the preprocessor are in `/script/data_preprocessor/`

- `drive_and_act_image_preprocessing.ipynb`, `syntheticcabin_ir_data_preprocessing.ipynb`, and `syntheticcabin_1m_preprocessor.ipynb` are used for such formatting.
- `fix_bbox_format.ipynb` - is a minor script for fixing inconsistant human bounding box annotation format.
- `syntheticcabin_1m_data_preprocessing.ipynb` is a script to preload all of the SyntheticCabin IR 1M into a pickle file.

## 2D Pose Estimation

All script for performing 2D Pose Estimation are in `script/pose_estimation_2d`. The folder contains a subfolder for each dataset.

- For SyntheticCabin IR, we have script for both finetuning and inference.
- For Drive&Act, we use only pretrained models for inference.
- For SyntheticCabin IR 1M, we use only annotated 2D/3D poses, so we don't need to perform any 2D pose estimation.

## 3D Pose Estimation Experiments

### Research Question 1 - Monocular 3D pose estimation on Synthetic Data

In `experiments/rq1_monocular_3d_pose_estimation_on_synthetic_data`, we have `run_experiment.py` to perform experiments for evalutate the possibility of using monocular 3D pose estimation on Synthetic Data. `result_eval.ipynb` is used for compiling the experiment results.

### Research Question 2 - Transfer Learning

In `experiments/rq2_transfer_learning`, we have each `run_experiment_*.py` for each type of configuration including using Drive&Act alone for trainning, using synthetic data for pretranining, and Drive&Act for finetuning, and, lastly, using the synthetic pretrained models directly without finetuning. In any case, we evaluate on testing set of Drive&Act.

### Research Question 3 - Weakly supervised learning

In `experiments/rq3_weakly_supervised`, we have multiple script for each dataset and each combination of viewpoints.
