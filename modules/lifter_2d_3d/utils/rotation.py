import numpy as np


def rotate2D_to_x_axis(line, points):
    # Extract the direction vector of the line
    direction = line[0] - line[1]

    # Calculate the angle of rotation
    angle = -np.arctan2(direction[1], direction[0])

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])

    # Apply the rotation to the line
    rotated_line = np.dot(rotation_matrix, points.T).T

    return rotated_line

# def rotate_line_to_x_axis(line):
#     # Extract the direction vector of the line
#     direction = line[1] - line[0]

#     # Calculate the angle for rotation
#     theta = -np.arctan2(direction[1], direction[0])

#     # Create the 2D rotation matrix
#     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                [np.sin(theta), np.cos(theta)]])

#     # Apply the rotation to the line
#     rotated_line = np.dot(rotation_matrix, line.T).T

#     return rotated_line

# def rotate3D_to_x_axis(line, points):
#     # Extract the direction vector of the line
#     direction = line[1] - line[0]

#     # Calculate the angles for rotations around the y and z axes
#     theta_y = -np.arctan2(direction[2], direction[0])
#     theta_z = np.arctan2(direction[1], np.sqrt(direction[0]**2 + direction[2]**2))

#     # Create the rotation matrices for y and z axes
#     rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
#                                   [0, 1, 0],
#                                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

#     rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                                   [np.sin(theta_z), np.cos(theta_z), 0],
#                                   [0, 0, 1]])

#     # Apply the rotations to the line
#     rotated_line = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, points.T)).T

#     return rotated_line
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
    results = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, points.T)).T

    return results
