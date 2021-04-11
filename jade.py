#!/usr/bin/env python
#######################################################################
# jadeR.py -- Blind source separation of real signals
#
# Version 1.8
#
# Copyright 2005, Jean-Francois Cardoso (Original MATLAB code)
# Copyright 2007, Gabriel J.L. Beckers (NumPy translation)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#######################################################################

# This file can be either used from the command line (type
# 'python jadeR.py --help' for usage, or see docstring of function main below)
# or it can be imported as a module in a python shell or program (use
# 'import jadeR').

# Comments in this source file are from the original MATLAB program, unless they
# are preceded by 'GB'.


"""
jadeR

This module contains only one function, jadeR, which does blind source
separation of real signals. Hopefully more ICA algorithms will be added
in the future.

jadeR requires NumPy.
"""
import numpy as np
from numpy.linalg import eig, pinv


def jadeR(X):
    """
    Blind separation of real signals with JADE.

    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm developed by Jean-Francois Cardoso.
    More information about JADE can be found among others in:
    Cardoso, J. (1999) High-order contrasts for independent component analysis.
    Neural Computation, 11(1): 157-192.
    Or look at the website: http://www.tsi.enst.fr/~cardoso/guidesepsou.html

    More information about ICA can be found among others in Hyvarinen A., Karhunen J., Oja E. (2001).
    Independent Component Analysis, Wiley. Or at the website http://www.cis.hut.fi/aapo/papers/IJCNN99_tutorialweb/

    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by Gabriel Beckers, http://gbeckers.nl .

    Parameters:
        X -- an n x T data matrix (n sensors, T samples). Must be a NumPy array or matrix.

        m -- number of independent components to extract.
            Output matrix B will have size m x n so that only m sources are extracted.
            This is done by restricting the operation of jadeR to the m first principal components.
            Defaults to None, in which case m == n.

        verbose -- print info on progress. Default is False.

    Returns:
        An m*n matrix B (NumPy matrix type), such that Y = B * X are separated
        sources extracted from the n * T data matrix X. If m is omitted, B is a
        square n * n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y = B * X.

    Quick notes (more at the end of this file):

    o This code is for REAL-valued signals.  A MATLAB implementation of JADE
        for both real and complex signals is also available from
        http://sig.enst.fr/~cardoso/stuff.html

    o This algorithm differs from the first released implementations of
        JADE in that it has been optimized to deal more efficiently
        1) with real signals (as opposed to complex)
        2) with the case when the ICA model does not necessarily hold.

    o There is a practical limit to the number of independent
        components that can be extracted with this implementation.  Note
        that the first step of JADE amounts to a PCA with dimensionality
        reduction from n to m (which defaults to n).  In practice m
        cannot be `very large` (more than 40, 50, 60... depending on
        available memory)

    o See more notes, references and revision history at the end of
        this file and more stuff on the WEB
        http://sig.enst.fr/~cardoso/stuff.html

    o For more info on NumPy translation, see the end of this file.

    o This code is supposed to do a good job!  Please report any
        problem relating to the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code: Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation: Gabriel Beckers <gabriel@gbeckers.nl>
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.

    # origtype = X.dtype  # float64

    X = np.matrix(X.astype(np.float64))  # create a matrix from a copy of X created as a float 64 array

    [n, T] = X.shape

    m = int(n)

    X -= X.mean(1)

    # whitening & projection onto signal subspace
    # -------------------------------------------

    # An eigen basis for the sample covariance matrix
    [D, U] = eig((X * X.T) / float(T))
    # Sort by increasing variances
    k = D.argsort()
    Ds = D[k]

    # The m most significant princip. comp. by decreasing variance
    PCs = np.arange(n - 1, n - m - 1, -1)

    # PCA
    # At this stage, B does the PCA on m components
    B = U[:, k[PCs]].T

    # --- Scaling ---------------------------------
    # The scales of the principal components
    scales = np.sqrt(Ds[PCs])
    B = np.diag(1. / scales) * B
    # Sphering
    X = B * X

    # We have done the easy part: B is a whitening matrix and X is white.

    del U, D, Ds, k, PCs, scales

    # NOTE: At this stage, X is a PCA analysis in m components of the real
    # data, except that all its entries now have unit variance. Any further
    # rotation of X will preserve the property that X is a vector of
    # uncorrelated components. It remains to find the rotation matrix such
    # that the entries of X are not only uncorrelated but also `as independent
    # as possible". This independence is measured by correlations of order
    # higher than 2. We have defined such a measure of independence which 1)
    # is a reasonable approximation of the mutual information 2) can be
    # optimized by a `fast algorithm" This measure of independence also
    # corresponds to the `diagonality" of a set of cumulant matrices. The code
    # below finds the `missing rotation " as the matrix which best
    # diagonalizes a particular set of cumulant matrices.

    # Estimation of Cumulant Matrices
    # -------------------------------

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T  # transpose data to (256, 3)
    # Dim. of the space of real symm matrices
    dimsymm = (m * (m + 1)) // 2  # 6
    # number of cumulant matrices
    nbcm = dimsymm  # 6
    # Storage for cumulant matrices
    CM = np.matrix(np.zeros([m, m * nbcm], dtype=np.float64))
    R = np.matrix(np.eye(m, dtype=np.float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]
    # # Temp for a cum. matrix
    # Qij = matrix(zeros([m, m], dtype=float64))
    # # Temp
    # Xim = zeros(m, dtype=float64)
    # # Temp
    # Xijm = zeros(m, dtype=float64)

    # I am using a symmetry trick to save storage. I should write a short note
    # one of these days explaining what is going on here.
    # will index the columns of CM where to store the cum. mats.
    Range = np.arange(m)  # [0 1 2]

    for im in range(m):
        Xim = X[:, im]
        Xijm = np.multiply(Xim, Xim)
        Qij = np.multiply(Xijm, X).T * X / float(T) - R - 2 * np.dot(R[:, im], R[:, im].T)
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = np.multiply(Xim, X[:, jm])
            Qij = np.sqrt(2) * np.multiply(Xijm, X).T * X / float(T) - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T
            CM[:, Range] = Qij
            Range = Range + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
    # m x m*nbcm array.

    # Joint diagonalization of the cumulant matrices
    # ==============================================

    V = np.matrix(np.eye(m, dtype=np.float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]

    # Diag = zeros(m, dtype=float64)  # [0. 0. 0.]
    On = 0.0
    Range = np.arange(m)  # [0 1 2]
    for im in range(nbcm):  # nbcm == 6
        Diag = np.diag(CM[:, Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (np.multiply(CM, CM).sum(axis=0)).sum(axis=0) - On
    # A statistically scaled threshold on `small" angles
    seuil = 1.0e-6 / np.sqrt(T)  # 6.25e-08
    # sweep number
    encore = True
    sweep = 0
    # Total number of rotations
    updates = 0
    # Number of rotations in a given seep

    # Joint diagonalization proper

    while encore:
        encore = False
        sweep += 1
        upds = 0

        for p in range(m - 1):  # m == 3
            for q in range(p + 1, m):  # p == 1 | range(p+1, m) == [2]

                Ip = np.arange(p, m * nbcm, m)  # [ 0  3  6  9 12 15] [ 0  3  6  9 12 15] [ 1  4  7 10 13 16]
                Iq = np.arange(q, m * nbcm, m)  # [ 1  4  7 10 13 16] [ 2  5  8 11 14 17] [ 2  5  8 11 14 17]

                # computation of Givens angle
                g = np.concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]  # -6.54012319852 4.44880758012 -1.96674621935
                toff = gg[0, 1] + gg[1, 0]  # -15.629032394 -4.3847687273 6.72969915184
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(
                    ton * ton + toff * toff))  # -0.491778606993 -0.194537202087 0.463781701868
                Gain = (np.sqrt(ton * ton + toff * toff) - ton) / 4.0  # 5.87059352069 0.449409565866 2.24448683877

                if np.abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = np.cos(theta)
                    s = np.sin(theta)
                    G = np.matrix([[c, -s], [s, c]])  # DON"T PRINT THIS! IT"LL BREAK THINGS! HELLA LONG
                    pair = np.array([p, q])  # don't print this either
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, np.concatenate([Ip, Iq])] = np.append(c * CM[:, Ip] + s * CM[:, Iq],
                                                                -s * CM[:, Ip] + c * CM[:, Iq],
                                                                axis=1)
                    On = On + Gain
                    Off = Off - Gain
        updates = updates + upds  # 3 6 9 9

    # A separating matrix
    # -------------------

    B = V.T * B
    # [[ 0.17242566  0.10485568 -0.7373937 ]
    #  [-0.41923305 -0.84589716  1.41050008]
    #  [ 1.12505903 -2.42824508  0.92226197]]

    # Permute the rows of the separating matrix B to get the most energetic
    # components first. Here the **signals** are normalized to unit variance.
    # Therefore, the sort is according to the norm of the columns of
    # A = pinv(B)

    A = pinv(B)
    # [[-3.35031851 -2.14563715  0.60277625]
    #  [-2.49989794 -1.25230985 -0.0835184 ]
    #  [-2.49501641 -0.67979249  0.12907178]]
    keys = np.array(np.argsort(np.multiply(A, A).sum(axis=0)[0]))[0]  # [2 1 0]
    B = B[keys, :]
    # [[ 1.12505903 -2.42824508  0.92226197]
    #  [-0.41923305 -0.84589716  1.41050008]
    #  [ 0.17242566  0.10485568 -0.7373937 ]]
    B = B[::-1, :]
    # [[ 0.17242566  0.10485568 -0.7373937 ]
    #  [-0.41923305 -0.84589716  1.41050008]
    #  [ 1.12505903 -2.42824508  0.92226197]]
    # just a trick to deal with sign == 0
    b = B[:, 0]  # [[ 0.17242566] [-0.41923305] [ 1.12505903]]
    signs = np.array(np.sign(np.sign(b) + 0.1).T)[0]  # [1. -1. 1.]
    B = np.diag(signs) * B
    # [[ 0.17242566  0.10485568 -0.7373937 ]
    #  [ 0.41923305  0.84589716 -1.41050008]
    #  [ 1.12505903 -2.42824508  0.92226197]]
    return B


def main(X):
    B = jadeR(X)
    Y = B * np.matrix(X)
    return Y.T

# B = B.astype(origtype)
# savetxt("ct_jade_data.txt", Y.T)
