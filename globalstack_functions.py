import numpy as np
from numba import jit
from numba import int64, float64
import math

from numba.experimental import jitclass

spec2 = [
    ('__nn', int64),
    ('__numel', int64),
    ('__u', int64),
    ('__uu', int64),
    ('__ul', int64),
    ('__left', int64[:]),
    ('__z', float64[:])
]


@jitclass(spec2)
class pq:
    def __init__(self, z):

        self.__nn = np.int64(len(z))
        self.__numel = np.int64(0)
        self.__u = np.int64(0)
        self.__uu = np.int64(0)
        self.__ul = np.int64(0)
        self.__left = np.full(len(z) + 1, 0)
        self.__z = np.concatenate((np.zeros(1).ravel(), z.ravel()))

    def top(self):
        return self.__left[1] - 1

    def get(self):
        return self.__z[self.__left]

    # @property
    def pop(self):
        self.__uu = self.__left[1]
        self.__left[1] = self.__left[self.__numel]
        self.__left[self.__numel] = 0
        self.__u = 2
        self.__ul = np.int(self.__u / 2)
        while self.__u < self.__numel - 2:
            if self.__z[self.__left[self.__u]] <= self.__z[self.__left[self.__u + 1]]:

                if self.__z[self.__left[self.__ul]] >= self.__z[self.__left[self.__u]]:
                    t = self.__left[self.__ul]
                    self.__left[self.__ul] = self.__left[self.__u]
                    self.__left[self.__u] = t

                    self.__ul = self.__u
                    self.__u *= 2
                else:
                    break

            elif self.__z[self.__left[self.__ul]] > self.__z[self.__left[self.__u + 1]]:

                t = self.__left[self.__ul]
                self.__left[self.__ul] = self.__left[self.__u + 1]
                self.__left[self.__u + 1] = t
                self.__u = 2 * (self.__u + 1)
                self.__ul = np.int(self.__u / 2)

            else:

                break

        self.__numel -= 1
        return self

    def push(self, i):

        i += 1
        self.__numel += 1

        self.__u = self.__numel
        self.__ul = np.int(self.__u / 2)

        self.__left[self.__u] = i
        while self.__ul > 0:
            if self.__z[self.__left[self.__ul]] >= self.__z[self.__left[self.__u]]:

                t = self.__left[self.__ul]
                self.__left[self.__ul] = self.__left[self.__u]
                self.__left[self.__u] = t

            else:
                break
            self.__u = np.int(self.__u / 2)
            self.__ul = np.int(self.__u / 2)
        return self


#
@jit(nopython=True)
def sinkfill(Z):
    c = int(0)
    ny, nx = np.shape(Z)
    nn = ny * nx
    p = int(0)
    closed = np.full(nn, False)
    idx = [1, -1, ny, -ny, -ny + 1, -ny - 1, ny + 1, ny - 1]
    open = pq(Z.transpose().flatten())

    for i in range(0, ny):
        for j in range(0, nx):
            if (i == 0) or (j == 0) or (j == nx - 1) or (i == nx - 1) or (Z[i,j] <=0):
                ij = j * ny + i
                if not closed[ij]:
                    closed[ij] = True
                    open = open.push(ij)
                    c += 1

    pit = np.zeros(nn)
    pittop = int(-9999)
    while (c > 0) or (p > 0):
        if (p > 0) and (c > 0) and (pit[p - 1] == -9999):
            s = open.top()
            open = open.pop()
            c -= 1
            pittop = -9999
        elif p > 0:
            s = int(pit[p - 1])
            pit[p - 1] = -9999
            p -= 1
            if pittop == -9999:
                si, sj = lind(s, ny)
                pittop = Z[si, sj]
        else:
            s = int(open.top())
            open = open.pop()
            c -= 1
            pittop = -9999

        for i in range(8):

            ij = idx[i] + s
            si, sj = lind(s, ny)
            ii, jj = lind(ij, ny)
            if (ii >= 0) and (jj >= 0) and (ii < ny) and (jj < nx) and not closed[ij]:
                closed[ij] = True

                if Z[ii, jj] <= Z[si, sj]:

                    Z[ii, jj] = Z[si, sj] + 1e-8

                    pit[p] = ij

                    p += 1
                else:
                    open = open.push(ij)
                    c += 1
            if np.mod(ij, 1e8) == 0:
                print(ij)
    return Z

@jit(nopython=True)
def h_flowdir(d):
    """
    Translate flow dir from hydrosheds into flow dir from simplem format
    :param d: flow direction grid from hydrosheds
    :return: sx and sy, the flow direction in x and y directions ( i.e. -1, 0 , 1)
    """
    ny, nx = np.shape(d)
    sy = np.zeros(np.shape(d), dtype=np.int8)  # slopes formatted correctly for simplem
    sx = np.zeros(np.shape(d), dtype=np.int8)
    for i in range(ny):
        for j in range(nx):
            d1 = d[i, j]
            if (d1 == 0) or (d1 == 255):  # 0 is outlet to ocean, 255 is internally drained pour point
                sx[i, j] = 0
                sy[i, j] = 0
            elif d1 == 1:
                sx[i, j] = 1
                sy[i, j] = 0
            elif d1 == 2:
                sx[i, j] = 1
                sy[i, j] = 1
            elif d1 == 4:
                sx[i, j] = 0
                sy[i, j] = 1
            elif d1 == 8:
                sx[i, j] = -1
                sy[i, j] = 1
            elif d1 == 16:
                sx[i, j] = -1
                sy[i, j] = 0
            elif d1 == 32:
                sx[i, j] = -1
                sy[i, j] = -1
            elif d1 == 64:
                sx[i, j] = 0
                sy[i, j] = -1
            elif d1 == 128:
                sx[i, j] = 1
                sy[i, j] = -1

    
    return sx, sy


@jit(nopython=True)
def lind(xy, n):
    """
    compute bilinear index from linear
    :param xy:  linear index
    :param n: ny or nx (depending on row-major or col-major indexing)
    :return:
    """
    # Compute linear index from 2 points
    x = math.floor(xy / n)
    y = xy % n
    return int(y), int(x)


@jit()
def stack(sx, sy):
    """
    takes the input flordirs sx sy and makes the topologically ordered
     stack of the stream network in O(n) time
    :param sx: x flow direction grid
    :param sy: y flow direction grid
    :return: topologically ordered stack, I
    """

    c = 0
    k = 0
    ny, nx = np.shape(sx)
    I = np.zeros(int(ny * nx), dtype=np.int64)
    for i in range(ny):
        for j in range(nx):

            ij = j * ny + i
            i2 = i
            j2 = j
            if (sx[i, j]) == 0 and sy[i, j] == 0:
                I[c] = ij
                c += 1

                while k < c < ny * nx - 1:
                    for i1 in range(-1, 2):
                        for j1 in range(-1, 2):
                            if 0 < j2 + j1 < nx - 1 and 0 < i2 + i1 < ny - 1:
                                ij2 = (j2 + j1) * ny + i2 + i1
                                recrx = sx[int(i2 + i1), int(j2 + j1)]
                                recry = sy[int(i2 + i1), int(j2 + j1)]
                                if ((recrx != 0) or (recry != 0)) and ((recrx + j1 == 0) and (recry + i1 == 0)):
                                    I[c] = ij2
                                    c += 1

                    k = k + 1
                    ij = I[k]
                    i2, j2 = lind(ij, ny)
    return I
