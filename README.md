# 3D Pose Estimation

This repository is a part of the Synthetic Cabin Project

## Project Structure

- `modules/`
  - `data_preprocessing/` - contains data preprocessor for each dataset. The main jobs of the proprocessor are to format input images and their annotation into an COCO-style dataset.
  - `human_detector/`
    - `config/`
    - `human_detector.py`
    - `main.py`
  - `pose_estimator_2d/`
  - `lifter_2d_3d/`
    - `dataset/`
      - `base_dataset.py` - used for Synthetic Cabin IR
      - `drive_and_act_keypoint_dataset.py` - used for Drive&Act Dataset
      - `gan_keypoint_dataset.py` - used specifically for the RepNet model.
      - `synthetic_cabin_ir_1m_dataset.py` - used for Synthetic Cabin IR 1M
    - `utils/`
    - `model/`
      - `common/`
      - `linear_model/` - From [Martinez et al. (2017)](https://arxiv.org/abs/1705.03098). Implementation from [motokimura's GitHub](https://github.com/motokimura/3d-pose-baseline-pytorch) is used.
      - `semgcn/` - Implementation from [Zhao et al. (2019)](https://github.com/garyzhao/SemGCN)
      - `graph_mlp/` - Implementation from [Li et al. (2022)](https://github.com/Vegetebird/GraphMLP)
      - `graformer/` - Implementation from [Zhao et al. (2022)](https://github.com/Graformer/GraFormer)
      - `jointformer/` - Implementation from [Lutz et al. (2022)](https://github.com/seblutz/JointFormer)
      - `repnet/` - Implement PyTorch version based on TensorFlow implementation from [Bastian Wandt's GitHub](https://github.com/bastianwandt/RepNet).
  - `experiments/`
  - `utils/`
