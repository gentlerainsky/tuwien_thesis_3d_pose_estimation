import random
import numpy as np
import plotly.express as px

# Optimally distinct colors
# generated from https://mokole.com/palette.html
# joint_colors = [
#     '#2f4f4f',
#     '#8b4513',
#     '#006400',
#     '#4b0082',
#     '#ff0000',
#     '#ffa500',
#     '#ffff00',
#     '#00ff00',
#     '#00fa9a',
#     '#00bfff',
#     '#0000ff',
#     '#ff00ff',
#     '#dda0dd',
#     '#ff1493',
#     '#ffe4b5',
# ]
joint_colors = [
    '#2f4f4f',
    '#7f0000',
    '#006400',
    '#000080',
    '#ff0000',
    '#ffff00',
    '#c71585',
    '#00ff00',
    '#00fa9a',
    '#00ffff',
    '#0000ff',
    '#ffa500',
    '#ff00ff',
    '#1e90ff',
    '#f0e68c',
]
joint_sides = [
    0, # 'M',
    1, # 'L',
    2, # 'R',
    1, # 'L',
    2, # 'R',
    1, # 'L',
    2, # 'R',
    1, # 'L',
    2, # 'R',
    1, # 'L',
    2, # 'R',
    1, # 'L',
    2, # 'R',
    1, # 'L',
    2, # 'R',           
]

def generate_connection_line(vals, valid_keypoints=None):
    L = 0
    C = 1
    R = 2
    connections = [
        (0, 1, L, 'nose_left_eye'), # nose & left_eye
        (0, 2, R, 'nose_right_eye'), # nose & right_eye
        (1, 2, C, 'left_right_eye'), # left & right eyes
        (1, 3, L, 'left_ear_left_eye'), # left ear & eye
        (2, 4, R, 'right_ear_right_eye'), # right ear & eye
        # (0, 5, L, 'nose_left_shoulder'), # nose & left shoulder
        # (0, 6, R, 'nose_right_shoulder'), # nose & right shoulder
        # (3, 5, L, 'left_ear_shoulder'), # left ear & shoulder
        # (4, 6, R, 'right_ear_shoulder'), # right ear & shoulder
        (5, 6, C, 'left_shoulder_right_sholder'), # left & right shoulder
        (5, 7, L, 'left_sholder_left_elbow'), # left shoulder & elbow
        (5, 11, L, 'left_shoulder_left_hip'), # left shoulder & hip
        (6, 8, R, 'right_shoulder_right_elbow'), # right shoulder & elbow
        (6, 12, R, 'right_shoulder_right_hip'), # right shoulder & hip
        (7, 9, L, 'left_elbow_left_wrist'), # left elbow & wrist
        (8, 10, R, 'right_elbow_right_wrist'), # right elbow & wrist
        (11, 12, C, 'left_hip_right_hip'), # left & right hip
        # (11, 13, L, 'left_hip_left_knee'), # left hip & knee
        # (12, 14, R, 'right_hip_right_knee'), # right hip & knee
        # (13, 15, L, 'left_knee_left_ankle'), # left knee & ankle
        # (14, 16, R, 'right_knee_right_ankle') # right knee & ankle
    ]
    connection_lines = []

    connection_count = 0
    for i, connection in enumerate(connections):
        if (valid_keypoints is not None) and ((connection[0] not in valid_keypoints) or (connection[1] not in valid_keypoints)):
            continue
        x, y, z = [np.array([vals[connection[0], j], vals[connection[1], j]]) for j in range(3)]
        for px, py, pz in zip(x, y, z):
            connection_lines.append({
                # "line": connection_count,
                "line": connection[3],
                "left_right": connection[2],
                "x": px,
                "y": py,
                "z": pz
            })
        connection_count += 1
    return connection_lines


def get_sample_from_loader(dataloader, index = None):
    loader = dataloader
    sample = None
    item_index = index
    if item_index is None:
        item_index = random.randint(0, len(loader.dataset))
    sample = loader.dataset[item_index]
    return item_index, sample


def visualize_pose(pose_df):
    fig = px.line_3d(pose_df, x="z", y="x", z="y", color="line")
    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed'},
            'zaxis': {'autorange': 'reversed'},
        }
    )
    fig.show()
