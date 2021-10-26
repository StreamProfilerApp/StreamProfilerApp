## These are functions directly related to querying elevation and calulating chi

import numpy as np
from numba import jit
import math


@jit(nopython=True)
def lind(xy, n):  # Compute linear index from 2 point
    x = math.floor(int(xy) / n)
    y = xy % n
    return int(y), int(x)


@jit(nopython=True)
def getstream(i, j, stackx, stacky):
    """
    Gets the indices from the receiver grids
    :param i: input initial index i
    :param j: input initial index j
    :param stackx: x receiver grid
    :param stacky: y receiver grid
    :return: linear index of stream locations on the grid
    """
    strm = np.zeros(0, dtype=np.int64)
    ny, nx = np.shape(stackx)
    while 1:
        ij = i + j * ny
        strm = np.append(strm, int(ij))
        i2 = i + np.int64(stacky[i, j])
        j2 = j + np.int64(stackx[i, j])
        i = i2
        j = j2
        if (stackx[i, j] == 0) and (stacky[i, j] == 0):
            break
        if len(strm) > 100000:
            break

    return strm


@jit(nopython=True)
def getstream_elev(i, j, stackx, stacky, dem, elev=0):
    """
    Gets the indices from the receiver grids, until a given elevation
    (this is redundant, one of these should be replaced in the future)
    :param i: input initial index i
    :param j: input initial index j
    :param stackx: x receiver grid
    :param stacky: y receiver grid
    :param elev: elevation above which to profile to
    :return: linear index of stream locations on the grid
    """
    #nd=nd[0]
    strm = np.zeros(0, dtype=np.int64)
    ny, nx = np.shape(stackx)
    z=np.zeros(0,dtype=np.float64)
    while 1:


        i2 = i + np.int64(stacky[i, j])
        j2 = j + np.int64(stackx[i, j])
        i = i2
        j = j2
        ij = i + j * ny
        # print(elev)
        zi = dem[0,i, j]
        if zi < elev:
            break
        strm = np.append(strm, int(ij))
        z = np.append(z,float(zi))
        
#         if dem[0, i, j] < elev:
#             break
        if (stackx[i, j] == 0) and (stacky[i, j] == 0) or (i == ny-2 or j == nx - 2 or j == 1 or i ==1):
            break
        if len(strm) > 100000:
            break


    return strm, z


@jit(nopython=True)
def get_upstream(i, j, stackx, stacky, recy, recx, acc):
    """
    Get the upstream nodes from a point. So far this is unused - maybe eventually ...
    :param i: Initial i value
    :param j: Initial j value
    :param stackx: stack x grid
    :param stacky: stack y grid
    :param recy: receiver x grid
    :param recx: receiver y grid
    :param acc: accumulation grid
    :return: linear indices along stream
    """
    strm = np.zeros(0, dtype=np.int64)
    ny, nx = np.shape(stackx)
    iacc = acc[i, j]
    while 1:
        if acc[i, j] > iacc:
            break
        ij = i + j * ny
        strm = np.append(strm, int(ij))
        i2 = i + np.int64(stacky[i, j])
        j2 = j + np.int64(stackx[i, j])
        print(i2, j2, iacc, acc[i, j])
        i = i2
        j = j2
        if (recy[i, j] == 0) and (recx[i, j] == 0):
            break
        if len(strm) > 100000:
            break

    return strm


def chicalc(A, dist, theta, U=1):
    """
    :param A: Accumulation (linear) along stream
    :param dist: distance (linear) along stream
    :param theta: aka concavity
    :param U: If we have variable uplift
    :return: linear chi values along stream
    """
    dx = dy = 90
    chi = np.cumsum(U / (np.flip(A[:-1]) ) ** theta  * -np.diff(np.flip(dist)))
    # chi[A<1e5] = np.nan
    return np.flip(chi)


@jit()
def getchi(I, s, dx):
    """

    :param I: Input linear stack
    :param s: Input linear receiver
    :param dx:
    :return: linear chi values along stream
    """
    A = np.ones(np.shape(s))
    chi = np.zeros(np.shape(s))
    dy = 90
    dx = 78
    ny, nx = np.shape(s)
    for ij in range(len(I) - 1, 0, -1):
        j = int(I[ij] / ny)
        i = I[ij] % ny
        j2 = int(I[ij] / ny)
        i2 = I[ij] % ny
        if I[ij] != s[i, j]:
            A[i2, j2] += A[i, j]
    for ij in range(0, len(I) - 1):
        i = I[ij] % ny
        j = int(I[ij] / ny)
        i2 = s[i, j] % ny
        j2 = int(s[i, j] / ny)
        if I[ij] != s[i, j]:
            chi[i, j] = chi[i2, j2] + 1 / A[i, j] ** .5 * np.sqrt(((i - i2) * dy) ** 2 + ((j - j2) * dx) ** 2)
        if ij % 100000 == 0:
            print(ij / len(I))
    return chi
