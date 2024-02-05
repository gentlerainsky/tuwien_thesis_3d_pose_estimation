import pickle
from modules.experiments.dataset import (
    construct_synthetic_cabin_ir,
    construct_drive_and_act_dataset,
    synthetic_cabin_ir_dataset_root_path,
    drive_and_act_dataset_root_path
)
from modules.lifter_2d_3d.model.linear_model.lit_linear_model import LitSimpleBaselineLinear
from modules.lifter_2d_3d.model.semgcn.lit_semgcn import LitSemGCN
from modules.lifter_2d_3d.model.graph_mlp.lit_graphmlp import LitGraphMLP
from modules.lifter_2d_3d.model.graformer.lit_graformer import LitGraformer
from modules.lifter_2d_3d.model.jointformer.lit_jointformer import LitJointFormer

ALL_LIGHTNING_MODELS = [
    LitSimpleBaselineLinear,
    LitSemGCN,
    LitGraphMLP,
    LitGraformer,
    LitJointFormer
]

SYNTHETIC_CABIN_VIEWPOINTS = [
    'A_Pillar_Codriver',
    'A_Pillar_Driver',
    'Rear_Mirror'
]

DRIVE_AND_ACT_VIEWPOINTS = [
    'a_column_co_driver',
    'a_column_driver',
    'inner_mirror'
]

# actors sampled with the following code
############################
# from pprint import pprint
# import random
# combinations = list(itertools.combinations(all_train_actors, 2))
# actor_samples = random.sample(combinations, 10)
############################
two_actors_samples = [
    ('vp2', 'vp6'),
    ('vp3', 'vp8'),
    ('vp1', 'vp2'),
    ('vp6', 'vp8'),
    ('vp1', 'vp5'),
    ('vp7', 'vp8'),
    ('vp1', 'vp7'),
    ('vp1', 'vp4'),
    ('vp3', 'vp7'),
    ('vp4', 'vp5')
]

four_actors_samples = [
    ('vp2', 'vp3', 'vp6', 'vp7'),
    ('vp3', 'vp4', 'vp6', 'vp8'),
    ('vp1', 'vp2', 'vp5', 'vp6'),
    ('vp2', 'vp3', 'vp5', 'vp7'),
    ('vp1', 'vp4', 'vp5', 'vp8'),
    ('vp1', 'vp2', 'vp6', 'vp8'),
    ('vp1', 'vp2', 'vp3', 'vp6'),
    ('vp2', 'vp5', 'vp6', 'vp8'),
    ('vp4', 'vp5', 'vp7', 'vp8'),
    ('vp2', 'vp3', 'vp4', 'vp7')
]

synthetic_cabin_ir_1m_preprocess_loaders = {
    # For ALPLab Dataset
    "Dashboard_Front_Front_Left_OMS_01": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/Dashboard_Front_Front_Left_OMS_01.pkl"
    },
    # For Co Driver Pillar
    "A_Pillar_Codriver_Front_Left_Front_TopLeft_Rear_Mirror": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/A_Pillar_Codriver_Front_Left_Front_TopLeft_Rear_Mirror.pkl"
    },
    # For Driver Pillar
    "A_Pillar_Driver_Front_Right_Front_TopRight": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/A_Pillar_Driver_Front_Right_Front_TopRight.pkl"
    },
    # For Rear Mirror
    "Dashboard_Front_OMS_01": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/Dashboard_Front_OMS_01.pkl"
    },
    # All views
    "all_views": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/all_views.pkl"
    },
}

driver_and_act_pretrained_map = {
    'a_column_co_driver': 'A_Pillar_Codriver_Front_Left_Front_TopLeft_Rear_Mirror',
    'a_column_driver': 'A_Pillar_Driver_Front_Right_Front_TopRight',
    'inner_mirror': 'Dashboard_Front_OMS_01'
}

def get_processed_synthetic_cabin_ir_1m_loaders(viewpoint_name):
    if type(viewpoint_name) is list:
        viewpoint_name = '_'.join(sorted(viewpoint_name))
    with open(synthetic_cabin_ir_1m_preprocess_loaders[viewpoint_name]['path'], 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_synthetic_cabin_ir_loaders(viewpoint_name):
    return construct_synthetic_cabin_ir(
        dataset_root_path=synthetic_cabin_ir_dataset_root_path,
        viewpoint=viewpoint_name
    )

def get_drive_and_act_loaders(viewpoint_name):
    return construct_drive_and_act_dataset(
        dataset_root_path=drive_and_act_dataset_root_path,
        viewpoint=viewpoint_name
    )
