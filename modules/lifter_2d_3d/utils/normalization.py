import numpy as np

def normalize_2d_pose_to_image(pose2d, image_width, image_height):
    out_pose2d = np.copy(pose2d)
    out_pose2d[:, 0] = out_pose2d[:, 0] / image_width
    out_pose2d[:, 1] = out_pose2d[:, 1] / image_height
    return out_pose2d, image_width, image_height


def normalize_2d_pose_to_bbox(pose2d, bbox, bbox_format):
    # scale by the bounding box
    # note that 3D keypoints is usually already scaled.
    x, y, w, h = bbox
    if bbox_format == 'xyxy':
        x, y, x2, y2 = bbox
        w = x2 - x
        h = y2 - y
    out_pose2d = np.copy(pose2d)
    out_pose2d[:, 0] = out_pose2d[:, 0] / w
    out_pose2d[:, 1] = out_pose2d[:, 1] / h
    return out_pose2d, w, h


def normalize_2d_pose_to_pose(pose2d):
    # scale by the max-min position of 2D poses
    x_max, y_max = np.max(pose2d[:, :2], axis=0)
    x_min, y_min = np.min(pose2d[:, :2], axis=0)
    w = x_max - x_min
    h = y_max - y_min
    out_pose2d = np.copy(pose2d)
    out_pose2d[:, 0] = out_pose2d[:, 0] / w
    out_pose2d[:, 1] = out_pose2d[:, 1] / h
    return out_pose2d, w, h


def center_pose2d_to_neck(pose2d, left_shoulder_index=5, right_shoulder_index=6):
    # center
    # use neck as the root. Neck is defined as the center of left/right shoulder
    out_pose2d = np.copy(pose2d)
    root_2d = (out_pose2d[left_shoulder_index, :2] + out_pose2d[right_shoulder_index, :2]) / 2
    out_pose2d[:, :2] = out_pose2d[:, :2] - root_2d
    return out_pose2d, root_2d


def center_pose3d_to_neck(pose3d, left_shoulder_index=5, right_shoulder_index=6):
    out_pose3d = np.copy(pose3d)
    root_3d = (out_pose3d[left_shoulder_index] + out_pose3d[right_shoulder_index]) / 2
    out_pose3d = out_pose3d - root_3d
    return out_pose3d, root_3d


def rotate2D_to_x_axis(line, points):
    # Extract the direction vector of the line
    direction = line[0] - line[1]

    # Calculate the angle of rotation
    angle = -np.arctan2(direction[1], direction[0])

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])

    # Apply the rotation to the line
    transform_points = np.dot(rotation_matrix, points.T).T

    return transform_points, rotation_matrix


def rotate3D_to_x_axis(line, points):
    # Translate the line to the origin
    translation_vector = -line[0]
    translated_line = line + translation_vector

    # Calculate the rotation matrix to align the line with the x-axis
    direction = translated_line[0] - translated_line[1]
    theta_y = np.arctan2(direction[2], direction[0])
    theta_z = -np.arctan2(direction[1], np.sqrt(direction[0]**2 + direction[2]**2))

    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                  [0, 1, 0],
                                  [-np.sin(theta_y), 0, np.cos(theta_y)]])

    rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                  [np.sin(theta_z), np.cos(theta_z), 0],
                                  [0, 0, 1]])

    # Apply the transformations
    transform_points = np.dot(
        rotation_matrix_z, np.dot(rotation_matrix_y, points.T)
    ).T

    return transform_points, rotation_matrix_y, rotation_matrix_z


def normalize_rotation(pose2d, pose3d, left_shoulder_index=5, right_shoulder_index=6):
    out_pose2d = np.copy(pose2d)
    out_pose2d[:, :2], rotation_matrix = rotate2D_to_x_axis(
        out_pose2d[left_shoulder_index: right_shoulder_index + 1, :2],
        out_pose2d[:, :2]
    )
    out_pose3d = np.copy(pose3d)
    out_pose3d, rotation_matrix_y, rotation_matrix_z = rotate3D_to_x_axis(
        out_pose3d[left_shoulder_index: right_shoulder_index + 1],
        out_pose3d
    )
    return out_pose2d, out_pose3d
