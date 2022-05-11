from scipy.special import factorial
import numpy as np
from numpy.linalg import svd, matrix_rank, det

def get_viewer_data(fragments=None, keypoints=None):
    data = {}
    if fragments:
        data["fragments"] = fragments

    if keypoints:
        data["keypoints"] = keypoints
    return data


def nchoosek(n, k) -> int:
    return factorial(n) / (factorial(n - k) * factorial(k))

def helmert_nd(X, Y, S_bnb, R_bnb, T_bnb):
    # RALIGN - Rigid alignment of two sets of points in k - dimensional
    # Euclidean space. Given two sets of points in
    # correspondence, this function computes the scaling,
    # rotation, and translation that define the transform TR
    # that minimizes the sum of squared errors between TR(X)
    # and its corresponding points in Y. This routine takes
    # O(nk ^ 3)-time.
    #
    # Inputs:
    # X - a kxn matrix whose columns are points
    # Y - a kxn matrix whose columns are points that correspond to
    # the points in X
    # Outputs:
    # c, R, t - the scaling, rotation matrix, and translation vector
    # defining the linear map TR as
    #
    # TR(x) = c * R * x + t
    #
    # such that the average norm of TR(X(:, i) - Y(:, i))
    # is minimized.
    #
    # See also:
    #
    # "Least-Squares Estimation of Transformation Parameters Between
    # Two Point Patterns."  Shinji Umeyama.  IEEE Transactions on
    # Pattern Analysis and Machine Intelligence.Vol. 13, No. 4,
    # April 1991.

    # Copyright(C) 2002 Mark A.Paskin
    #
    # This program is free software; you can redistribute it and / or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation; either version 2 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful, but
    # WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    # General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License
    # along with this program; if not, write to the Free Software
    # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111 - 1307
    # USA.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    X = np.transpose(X)
    Y = np.transpose(Y)
    (m, n) = X.shape

    mx = np.mean(X, axis=1) # Eqn.(34)
    my = np.mean(Y, axis=1) # Eqn.(35)

    Xc = X - np.tile(mx, (1, n)).reshape(X.shape)
    Yc = Y - np.tile(my, (1, n)).reshape(Y.shape)

    sx = np.mean(np.sum(np.power(Xc, 2), axis=0)) # Eqn.(36)
    sy = np.mean(np.sum(np.power(Yc, 2), axis=0)) # Eqn.(37)

    Sxy = np.matmul(Yc, np.transpose(Xc))/n  # Eqn. (38)

    U, D, V = svd(Sxy)

    r = matrix_rank(Sxy)
    d = det(Sxy)

    S = np.eye(m)
    if (r > m - 1):
        if (det(Sxy) < 0):
            S[m-1, m-1] = -1
    elif(r == m - 1):
        if (det(U) * det(V) < 0):
            S[m-1, m-1] = -1
    else:
        print('Insufficient rank in covariance to determine rigid transform. Returning input values for R, T, s.');
        R = R_bnb
        c = S_bnb
        t = T_bnb
        return (R, c, t)

    SV = np.matmul(S, np.transpose(V))
    R = np.matmul(U, SV) # Eqn. (40)
    c = np.trace(np.matmul(np.diag(D), S))/sx # Eqn.(42)
    tmp = np.matmul(R, mx)
    t = my - (c * tmp) # Eqn.(41)

    return (R, c, t)


