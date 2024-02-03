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
    "A_Pillar_Codriver_Front_Left_Front_TopLeft_Rear_Mirror": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/A_Pillar_Codriver_Front_Left_Front_TopLeft_Rear_Mirror.pkl"
    },
    "A_Pillar_Driver_Front_Right_Front_TopRight": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/A_Pillar_Driver_Front_Right_Front_TopRight.pkl"
    },
    "Dashboard_Front_OMS_01": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/Dashboard_Front_OMS_01.pkl"
    },
    "all_views": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/all_views.pkl"
    },
    "Dashboard_Front_Front_Left_OMS_01": {
        "path": "/root/data/processed/synthetic_cabin_1m/all_views/dataloaders/Dashboard_Front_Front_Left_OMS_01.pkl"
    },
}

import pickle

def get_processed_synthetic_cabin_ir_1m_loaders(viewpoint_name):
    if type(viewpoint_name) is list:
        viewpoint_name = '_'.join(sorted(viewpoint_name))
    with open(synthetic_cabin_ir_1m_preprocess_loaders[viewpoint_name]['path'], 'rb') as f:
        obj = pickle.load(f)
    return obj
