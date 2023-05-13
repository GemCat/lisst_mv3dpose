import cv2
import numpy as np
from numba import vectorize, float32, float64, jit, boolean
from math import sqrt


@vectorize([float64(float64,float64,float64,float64,float64)])
def line_to_point_distance(a,b,c,x,y):
    return abs(a*x + b*y + c) / sqrt(a**2 + b**2)


def get_projection_matrix(K, rvec, tvec):
    """
    generate the projection matrix from its sub-elements
    :param K: camera matirx
    :param rvec: rodrigues vector
    :param tvec: loc vector
    :return:
    """
    Rt = np.zeros((3, 4))
    Rt[:, 0:3] = rvec
    Rt[:, 3] = tvec[:, 0]

    return K @ Rt


def from_homogeneous(x):
    """
    convert from homogeneous coordinate
    :param x:
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if len(x.shape) == 1:
        h = x[-1]
        if h != 0:
            x = x/h
            return x[0:-1]
        else:
            return None
    else:
        assert len(x.shape) == 2
        h = np.expand_dims(x[:, -1], axis=1)
        x = x / h
        return x[:, 0:-1]


def reproject_points_to_2d(pts3d, rvec, tvec, K, w, h,
                           distCoef = np.zeros((5, 1)),binary_mask=False):
    """
    :param pts3d:
    :param rvec:
    :param tvec:
    :param K:
    :param w:
    :param h:
    :param distCoef:to match OpenCV API
    :return:
    """
    if len(pts3d) == 0:
        return [], []
    Pts3d = pts3d.astype('float32')
    pts2d, _ = cv2.projectPoints(Pts3d, rvec, tvec, K, distCoef)
    pts2d = np.squeeze(pts2d)
    if len(pts2d.shape) == 1:
        pts2d = np.expand_dims(pts2d, axis=0)

    x = pts2d[:, 0]
    y = pts2d[:, 1]

    mask = (x > 0) * 1
    mask *= (x < w) * 1
    mask *= (y > 0) * 1
    mask *= (y < h) * 1

    if not binary_mask:
        mask = np.nonzero(mask)

    return pts2d, mask