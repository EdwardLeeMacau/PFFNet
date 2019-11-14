"""
  FileName     [ functions.py ]
  PackageName  [ DarkChannelPrior ]
  Synophsis    [ ]
"""
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy.ndimage as ndimage
import scipy.sparse.linalg as spla
from scipy import misc
from scipy.sparse import *

# TODO: Same as readIm.py
def readIm(im):
    return cv2.imread(im, cv2.IMREAD_COLOR)

# TODO: Same as darkChannel.py
def getDarkChannel(im, numWindowPixels=15):
    """ Return J^{dark} """
    height, width, _ = im.shape
    padding = math.ceil(numWindowPixels / 2.0)

    J = np.zeros((height,width))
    paddedImage = np.pad(im, (padding, padding), 'constant', constant_values=(np.inf, np.inf))

    for j in range(0, height):
        for i in range(0, width):
            window = paddedImage[j : j + numWindowPixels - 1, i : i + numWindowPixels - 1, :]
            J[j, i] = np.amin(window)

    return J

# TODO: Same as atmLight.py 
def getAtmLight(im, JDark, top=0.001):
    height, width, _ = im.shape
    totalPixels = width * height
    
    ImVec = np.reshape(im, (totalPixels, 3))
    indices = np.argsort(np.reshape(JDark, (totalPixels, 1)), axis=0).flatten()

    # Pick the Top pixels, with the ratio **top**
    topPixels = math.floor(totalPixels * top)
    indices = indices[-topPixels:]

    # Mean Mode
    tempAtm = np.zeros((1,3))
    for ind in range(0, int(topPixels)):
        tempAtm = tempAtm + ImVec[indices[ind], :]

    A = tempAtm / topPixels

    return A.flatten()

# TODO： Same as transmissonEstimate.py
def getTransmission(im, A, omega=0.95):
    newImage = np.zeros(im.shape)

    for ind in range(0, 3):
        newImage[:, :, ind] = im[:, :, ind] / A[ind]

    return 1 - omega * getDarkChannel(newImage)

# TODO: Same as getRadiance.py
def getRadiance(atmLight, im, transmission, t0=0.1):
    J = np.zeros(im.shape)

    for ind in range(0, 3):
        J[:, :, ind] = atmLight[ind] + (im[:, :, ind] - atmLight[ind]) / np.maximum(transmission, t0)

    return J / np.amax(J)

# TODO: Same as matte.py
neighbors = []
neighbor_count = 0;

def performSoftMatting(im, transmission, windowRadius=1, numWindowPixels=9, 
                       windowWidth=3, epsilon=10**(-8), _lambda=10**(-4)):
    global neighbors

    width, height, _ = im.shape

    totalPixels = numWindowPixels ** 2
    windowIndicies = np.reshape(range(1, width * height + 1), (width, height), order='F')
    totalElements = totalPixels * (width - 2) * (height - 2)
    xIndicies = np.ones((1, totalElements))
    yIndicies = np.ones((1, totalElements))
    laplacian = np.zeros((1, totalElements))
    count = 0

    neighbors = np.empty((width * height, numWindowPixels))
    footprint = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])
    print(footprint.dtype)
    footprint = np.ones((3, 3))
    print(footprint.dtype)

    ndimage.generic_filter(windowIndicies, getWindow, footprint=footprint)

    U = epsilon / numWindowPixels * identity(windowWidth)

    for i in range(windowRadius, height - windowRadius):
        for j in range(windowRadius, width - windowRadius):
            window = im[j - windowRadius: j + windowRadius + 1, i - windowRadius : i + windowRadius + 1, :]

            reshapedWindow = np.reshape(window, (numWindowPixels, 3), order='F')

            diffFromMean = reshapedWindow.T - np.tile(np.mean(reshapedWindow, axis=0).T, (numWindowPixels, 1)).T

            window_covariance = np.dot(diffFromMean, diffFromMean.T) / numWindowPixels

            entry = identity(numWindowPixels) - (1 + np.dot(np.dot(diffFromMean.T, np.linalg.inv(window_covariance + U)), diffFromMean)) / float(numWindowPixels)

            temp = count * totalPixels
            temp2 = count * totalPixels + totalPixels

            newIndicies = np.tile(np.reshape(np.reshape(neighbors[height * j + i], (windowWidth, windowWidth)), (1, numWindowPixels), order='F'), (numWindowPixels, 1))

            xIndicies[0][temp : temp2] = newIndicies.flatten(1)
            yIndicies[0][temp : temp2] = (newIndicies.T).flatten(1)
            laplacian[0][temp : temp2] = entry.flatten(1)
            count += 1

    # L = csc_matrix((laplacian.flatten(), (xIndicies.flatten(), yIndicies.flatten())))
    L = csc_matrix((laplacian.flatten(), (xIndicies.flatten().astype(np.int), yIndicies.flatten().astype(np.int))))
    tBar = np.append(np.reshape(transmission.T, (width * height, 1)), [0])

    T = spla.spsolve(L + _lambda * identity(L.shape[0]), tBar * _lambda) 
    return np.reshape(np.delete(T, len(T) - 1), transmission.shape, order='F')

def getWindow(values):
    global neighbors, neighbor_count
    neighbors[neighbor_count] = np.reshape([values], (1,9))
    neighbor_count += 1

    return 0
