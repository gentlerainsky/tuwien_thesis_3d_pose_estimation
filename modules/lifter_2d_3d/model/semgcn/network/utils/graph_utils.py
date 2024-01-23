from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


connections = [
    (0, 1, "nose_left_eye"),  # nose & left_eye
    (0, 2, "nose_right_eye"),  # nose & right_eye
    (1, 2, "left_right_eye"),  # left & right eyes
    (1, 3, "left_eye_left_ear"),  # left eye & ear
    (2, 4, "right_eye_right_ear"),  # right eye & ear
    (0, 5, "nose_left_shoulder"),  # nose & left shoulder
    (0, 6, "nose_right_shoulder"),  # nose & right shoulder
    (3, 5, "left_ear_shoulder"),  # left ear & shoulder
    (4, 6, "right_ear_shoulder"),  # right ear & shoulder
    (5, 6, "left_shoulder_right_sholder"),  # left & right shoulder
    (5, 7, "left_sholder_left_elbow"),  # left shoulder & elbow
    (5, 11, "left_shoulder_left_hip"),  # left shoulder & hip
    (6, 8, "right_shoulder_right_elbow"),  # right shoulder & elbow
    (6, 12, "right_shoulder_right_hip"),  # right shoulder & hip
    (7, 9, "left_elbow_left_wrist"),  # left elbow & wrist
    (8, 10, "right_elbow_right_wrist"),  # right elbow & wrist
    (11, 12, "left_hip_right_hip"),  # left & right hip
    (11, 13, "left_hip_left_knee"),  # left hip & knee
    (12, 14, "right_hip_right_knee"),  # right hip & knee
    (13, 15, "left_knee_left_ankle"),  # left knee & ankle
    (14, 16, "right_knee_right_ankle"),  # right knee & ankle
]

parents = [
    -1,
    0,
    0,
    1,
    2,
    0,
    0,
    5,
    6,
    7,
    8,
    5,
    6,
    11,
    12,
    # 13,
    # 14
]

connections = np.array(connections)[:, :2].astype(int).tolist()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)
