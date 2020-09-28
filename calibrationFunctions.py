# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2

def invertTransformationMatrix(inMat):
    out = np.zeros((4, 4), np.float64)
    out[:3, :3] = inMat[:3, :3].T
    out[0, 3] = - np.sum(inMat[:3, 0] * inMat[:3, 3])
    out[1, 3] = - np.sum(inMat[:3, 1] * inMat[:3, 3])
    out[2, 3] = - np.sum(inMat[:3, 2] * inMat[:3, 3])
    out[3, 3] = 1
    return out


def invertTransformationMatrices(inMats):
    invMats = np.zeros_like(inMats)
    for idx in range(inMats.shape[2]):
        invMats[:, :, idx] = invertTransformationMatrix(inMats[:, :, idx])
    return invMats

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def transformationMatrixFromVecs(rVec, tVec):
    mat = np.zeros((4, 4))
    mat[:3, :3], _ = cv2.Rodrigues(rVec)
    mat[:3, 3] = tVec
    mat[3, 3] = 1
    return mat
"""
The following functions from 
https://github.com/eayvali/Pose-Estimation-for-Sensor-Calibration
"""

def rigid_registration(A, X, B):
    # nxnx4
    """solves for Y in YB=AX
    A: (4x4xn)
    B: (4x4xn)
    X= (4x4)
    Y= (4x4)
    n number of measurements
    ErrorStats: Registration error (mean,std)
    """
    n = A.shape[2];
    AX = np.zeros(A.shape)
    AXp = np.zeros(A.shape)
    Bp = np.zeros(B.shape)
    pAX = np.zeros(B[0:3, 3, :].shape)  # To calculate reg error
    pYB = np.zeros(B[0:3, 3, :].shape)  # To calculate reg error
    Y_est = np.eye(4)

    ErrorStats = np.zeros((2, 1))

    for ii in range(n):
        AX[:, :, ii] = np.matmul(A[:, :, ii], X)

        # Centroid of transformations t and that
    t = 1 / n * np.sum(AX[0:3, 3, :], 1);
    that = 1 / n * np.sum(B[0:3, 3, :], 1);
    AXp[0:3, 3, :] = AX[0:3, 3, :] - np.tile(t[:, np.newaxis], (1, n))
    Bp[0:3, 3, :] = B[0:3, 3, :] - np.tile(that[:, np.newaxis], (1, n))

    [i, j, k] = AX.shape;  # 4x4xn
    # Convert AX and B to 2D arrays
    AXp_2D = AXp.reshape((i, j * k))  # now it is 4x(4xn)
    Bp_2D = Bp.reshape((i, j * k))  # 4x(4xn)
    # %Calculates the best rotation
    U, S, Vt = np.linalg.svd(np.matmul(Bp_2D[0:3, :], AXp_2D[0:3, :].T))  # v is v' in matlab
    R_est = np.matmul(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R_est) < 0:
        print('Warning: Y_est returned a reflection')
        R_est = np.matmul(Vt.T, np.matmul(np.diag([1, 1, -1]), U.T))
        # Calculates the best transformation
    t_est = t - np.dot(R_est, that)
    Y_est[0:3, 0:3] = R_est
    Y_est[0:3, 3] = t_est
    # Calculate registration error
    pYB = np.matmul(R_est, B[0:3, 3, :]) + np.tile(t_est[:, np.newaxis], (1, n))  # 3xn
    pAX = AX[0:3, 3, :]
    Reg_error = np.linalg.norm(pAX - pYB, axis=0)  # 1xn
    ErrorStats[0] = np.mean(Reg_error)
    ErrorStats[1] = np.std(Reg_error)
    return Y_est, ErrorStats

def pose_estimation(A, B):
    n = A.shape[2];
    T = np.zeros([9, 9]);
    X_est = np.eye(4)
    Y_est = np.eye(4)

    # Permutate A and B to get gross motions
    idx = np.random.permutation(n)
    A = A[:, :, idx];
    B = B[:, :, idx];

    for ii in range(n - 1):
        Ra = A[0:3, 0:3, ii]
        Rb = B[0:3, 0:3, ii]
        #  K[9*ii:9*(ii+1),:] = np.concatenate((np.kron(Rb,Ra), -np.eye(9)),axis=1)
        T = T + np.kron(Rb, Ra);

    U, S, Vt = np.linalg.svd(T)
    xp = Vt.T[:, 0]
    yp = U[:, 0]
    X = np.reshape(xp, (3, 3), order="F")  # F: fortran/matlab reshape order
    Xn = (np.sign(np.linalg.det(X)) / np.abs(np.linalg.det(X)) ** (1 / 3)) * X
    # re-orthogonalize to guarantee that they are indeed rotations.
    U_n, S_n, Vt_n = np.linalg.svd(Xn)
    X = np.matmul(U_n, Vt_n)

    Y = np.reshape(yp, (3, 3), order="F")  # F: fortran/matlab reshape order
    Yn = (np.sign(np.linalg.det(Y)) / np.abs(np.linalg.det(Y)) ** (1 / 3)) * Y
    U_yn, S_yn, Vt_yn = np.linalg.svd(Yn)
    Y = np.matmul(U_yn, Vt_yn)

    A_est = np.zeros([3 * n, 6])
    b_est = np.zeros([3 * n, 1])
    for ii in range(n - 1):
        A_est[3 * ii:3 * ii + 3, :] = np.concatenate((-A[0:3, 0:3, ii], np.eye(3)), axis=1)
        b_est[3 * ii:3 * ii + 3, :] = np.transpose(
            A[0:3, 3, ii] - np.matmul(np.kron(B[0:3, 3, ii].T, np.eye(3)), np.reshape(Y, (9, 1), order="F")).T)

    t_est_np = np.linalg.lstsq(A_est, b_est, rcond=None)
    if t_est_np[2] < A_est.shape[1]:  # A_est.shape[1]=6
        print('Rank deficient')
    t_est = t_est_np[0]
    X_est[0:3, 0:3] = X
    X_est[0:3, 3] = t_est[0:3].T
    Y_est[0:3, 0:3] = Y
    Y_est[0:3, 3] = t_est[3:6].T
    # verify Y_est using rigid_registration
    Y_est_check, ErrorStats = rigid_registration(A, X_est, B)
    return X_est, Y_est, Y_est_check, ErrorStats
