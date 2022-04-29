import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
import math
import time
import concurrent
import numpy as np

from itertools import repeat

from Core.IO.serializedObjectIO import saveSerializedObjects, loadDataStructure
from Core.Data.DynamicData.breathingSignals import SyntheticBreathingSignal
from Core.Processing.DeformableDataAugmentationToolBox.generateDynamicSequencesFromModel import generateDeformationListFromBreathingSignalsAndModel
from Core.Processing.DeformableDataAugmentationToolBox.modelManipFunctions import *
from Core.Processing.DRRToolBox import forwardProjection
from Core.Processing.ImageProcessing.image2DManip import getBinaryMaskFromROIDRR, get2DMaskCenterOfMass
from Core.Processing.ImageProcessing.crop3D import *

def deformImageAndMaskAndComputeDRRs(img, ROIMask, deformation, projectionAngle=0, projectionAxis='Z', tryGPU=True, outputSize=[]):
    """
    This function is specific to this example and used to :
    - deform a CTImage and an ROIMask,
    - create DRR's for both,
    - binarize the DRR of the ROIMask
    - compute its center of mass
    """

    print('Start deformations and projections for deformation', deformation.name)
    image = deformation.deformImage(img, fillValue='closest', outputType=np.int16, tryGPU=tryGPU)
    # print(image.imageArray.shape, np.min(image.imageArray), np.max(image.imageArray), np.mean(image.imageArray))
    mask = deformation.deformImage(ROIMask, fillValue='closest', outputType=np.int16, tryGPU=tryGPU)

    DRR = forwardProjection(image, projectionAngle, axis=projectionAxis)
    DRRMask = forwardProjection(mask, projectionAngle, axis=projectionAxis)

    halfDiff = int((DRR.shape[1] - image.gridSize[2])/2)           ## not sure this will work if orientation is changed
    croppedDRR = DRR[:, halfDiff + 1:DRR.shape[1] - halfDiff - 1]         ## not sure this will work if orientation is changed
    croppedDRRMask = DRRMask[:, halfDiff + 1:DRRMask.shape[1] - halfDiff - 1] ## not sure this will work if orientation is changed

    if outputSize:
        # print('Before resampling')
        # print(croppedDRR.shape, np.min(croppedDRR), np.max(croppedDRR), np.mean(croppedDRR))
        ratio = [outputSize[0]/croppedDRR.shape[0], outputSize[1]/croppedDRR.shape[1]]
        croppedDRR = zoom(croppedDRR, ratio)
        croppedDRRMask = zoom(croppedDRRMask, ratio)
        # print('After resampling')
        # print(croppedDRR.shape, np.min(croppedDRR), np.max(croppedDRR), np.mean(croppedDRR))

    binaryDRRMask = getBinaryMaskFromROIDRR(croppedDRRMask)
    centerOfMass = get2DMaskCenterOfMass(binaryDRRMask)
    # print('CenterOfMass:', centerOfMass)

    del image  # to release the RAM
    del mask  # to release the RAM

    print('Deformations and projections finished for deformation', deformation.name)

    # plt.figure()
    # plt.subplot(1, 5, 1)
    # plt.imshow(DRR)
    # plt.subplot(1, 5, 2)
    # plt.imshow(croppedDRR)
    # plt.subplot(1, 5, 3)
    # plt.imshow(DRRMask)
    # plt.subplot(1, 5, 4)
    # plt.imshow(croppedDRRMask)
    # plt.subplot(1, 5, 5)
    # plt.imshow(binaryDRRMask)
    # plt.show()

    return [croppedDRR, binaryDRRMask, centerOfMass]