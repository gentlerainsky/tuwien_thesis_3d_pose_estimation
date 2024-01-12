import random
import numpy as np
import plotly.express as px
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Optimally distinct colors
# generated from https://mokole.com/palette.html
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


def plot_images(
        img_ids, img_paths, gt_kp_2d_list, figsize, colors,
        root_2d_list, scale_factor_list, bbox_list
    ):
    if len(img_paths) > 1:
        fig, axes = plt.subplots(1, len(img_paths), figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    for idx, (gt_kp_2d, img_id, img_path, root_2d, scale_factor, bbox) in enumerate(zip(gt_kp_2d_list, img_ids, img_paths, root_2d_list, scale_factor_list, bbox_list)):
        x_offset = root_2d[0]
        y_offset = root_2d[1]
        width, height = scale_factor
        axes[idx].scatter(
            gt_kp_2d[:, 0] * height + x_offset,
            gt_kp_2d[:, 1] * width + y_offset,
            c=colors,
            marker='o',
            alpha=.7,
        )
        axes[idx].imshow(plt.imread(img_path))
        # draw highest match bbox as green
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none', label='high bbox')
        axes[idx].add_patch(rect)
        axes[idx].set_title(f'Image {img_id}')

def plot_skeleton(
        gt_kp_3d_list,
        kp_3d_list,
        valids,
        figsize,
        colors,
        is_plot_gt_skeleton,
        elev=15.,
        azim=-120 + 45//2,
        roll=0
    ):
    fig = plt.figure(figsize=figsize)
    axes = []
    for idx, (gt_kp_3d, kp_3d, valid) in enumerate(zip(gt_kp_3d_list, kp_3d_list, valids)):
        if len(gt_kp_3d_list) == 1:
            axes = [plt.axes(projection='3d')]
        else:
            axes.append(fig.add_subplot(1, len(gt_kp_3d_list), idx + 1, projection='3d'))
        depth = gt_kp_3d[valid, 2]
        x = gt_kp_3d[valid, 0]
        y = gt_kp_3d[valid, 1]
        # plotting
        valid_colors = list(np.array(colors)[valid])
        gt_plot = axes[idx].scatter3D(x, depth, -y, c=valid_colors, marker='o', alpha=.7, depthshade=False, label='Label')
        depth = kp_3d[:, 2]
        x = kp_3d[:, 0]
        y = kp_3d[:, 1]
        predict_plot = axes[idx].scatter3D(x, depth, -y, c=colors, marker='x', depthshade=False, label='Predict')

        if is_plot_gt_skeleton:
            results = generate_connection_line(gt_kp_3d)
        else:
            results = generate_connection_line(kp_3d)
        for i in range(len(results) // 2):
            axes[idx].plot3D(
                [results[2*i]['x'], results[2*i + 1]['x']],
                [results[2*i]['z'], results[2*i + 1]['z']],
                [-results[2*i]['y'], -results[2*i + 1]['y']],
            )
        axes[idx].legend(loc='lower center')
        # axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Depth')
        # axes[idx].set_zlabel('Y')
        # axes[idx].set_xlim([-0.5, -0.5 + 0.8])
        # axes[idx].set_zlim([-0.5, -0.5 + 0.8])
        # axes[idx].set_ylim([0.7, 0.7 + 0.8])
        axes[idx].axes.set_aspect('equal')
        # axes[idx].xaxis.set_tick_params(labelsize=8, ticks=None)
        # axes[idx].yaxis.set_tick_params(labelsize=8, ticks=None)
        axes[idx].xaxis.set_ticks([])
        axes[idx].zaxis.set_ticks([])
        axes[idx].yaxis.set_tick_params(labelsize=8)
        axes[idx].view_init(elev=elev, azim=azim, roll=roll)
        axes[idx].xaxis.set_pane_color((0, 0, 0, .4))
        axes[idx].yaxis.set_pane_color((0, 0, 0, .4))
        axes[idx].zaxis.set_pane_color((0, 0, 0, .4))

def plot_samples(
        dataset_root_path,
        model,
        dataloader,
        data_subset,
        img_figsize,
        plot_figsize,
        sample_idices,
        is_plot_gt_skeleton=True
    ):
    model.eval()
    img_ids = []
    img_paths = []
    gt_keypoints_2d_list = []
    gt_keypoints_3d_list = []
    keypoints_3d_list = []
    root_2d_list = []
    root_3d_list = []
    scale_factor_list = []
    bbox_list = []
    valids = []
    for sample_idx in sample_idices:
        sample = dataloader.dataset.samples[sample_idx]
        gt_keypoints_3d = sample['keypoints3D']
        gt_keypoints_2d = sample['keypoints2D']
        root_2d = sample['root_2d']
        root_3d = sample['root_3d']
        bbox = sample['bbox']
        print(bbox)
        scale_factor = sample['scale_factor']
        valid = sample['valid']
        estimated_pose = model(torch.flatten(torch.tensor(sample['keypoints2D'][:, :2])).unsqueeze(0).float().to(model.device))
        keypoints_3d = estimated_pose[0].cpu().reshape([-1, 3]).detach().numpy()
    
        img_path = (dataset_root_path / 'images' / data_subset / sample['filenames']).as_posix()
        img_id = sample['id']
        img_ids.append(img_id)
        img_paths.append(img_path)
        gt_keypoints_3d_list.append(gt_keypoints_3d)
        gt_keypoints_2d_list.append(gt_keypoints_2d)
        keypoints_3d_list.append(keypoints_3d)
        valids.append(valid)
        root_2d_list.append(root_2d)
        root_3d_list.append(root_3d)
        scale_factor_list.append(scale_factor)
        bbox_list.append(bbox)
    num_joints = gt_keypoints_2d.shape[0]

    plot_images(
        img_ids=img_ids,
        img_paths=img_paths,
        gt_kp_2d_list=gt_keypoints_2d_list,
        root_2d_list=root_2d_list,
        scale_factor_list=scale_factor_list,
        bbox_list=bbox_list,
        figsize=img_figsize,
        colors=joint_colors[:num_joints]
    )
    plot_skeleton(
        gt_kp_3d_list=gt_keypoints_3d_list,
        kp_3d_list=keypoints_3d_list,
        valids=valids,
        figsize=plot_figsize,
        colors=joint_colors[:num_joints],
        is_plot_gt_skeleton=is_plot_gt_skeleton,
        elev=10., azim=-90 + 45//2, roll=0
    )
    plot_skeleton(
        gt_kp_3d_list=gt_keypoints_3d_list,
        kp_3d_list=keypoints_3d_list,
        valids=valids,
        figsize=plot_figsize,
        colors=joint_colors[:num_joints],
        is_plot_gt_skeleton=is_plot_gt_skeleton,
        elev=0, azim=0
    )