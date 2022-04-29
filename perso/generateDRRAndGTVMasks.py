"""
This file contains an example on how to:
- read model + ROI data from a serialized file
- create a breathing signal using the motion amplitude present in the model
- chose an ROI to apply the breathing signal to its center of mass
-

!!! does not work with public data for now since there is no struct in the public data !!!
"""
import sys 
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
import math
import time
import concurrent
import socket 
import multiprocessing as mp

print(os.getcwd())
os.chdir("/export/home/grotsartdehe/opentps")
#os.chdir("C:/Users/grotsartdehe/Documents/opentps")
print(socket.gethostname())
print(os.getcwd())
from itertools import repeat
from Core.IO.serializedObjectIO import saveSerializedObjects, loadDataStructure
from Core.Data.DynamicData.breathingSignals import SyntheticBreathingSignal
from Core.Processing.DeformableDataAugmentationToolBox.generateDynamicSequencesFromModel import generateDeformationListFromBreathingSignalsAndModel
from Core.Processing.DeformableDataAugmentationToolBox.modelManipFunctions import *
from Core.Processing.DRRToolBox import forwardProjection
from Core.Processing.ImageProcessing.image2DManip import getBinaryMaskFromROIDRR, get2DMaskCenterOfMass
from Core.Processing.ImageProcessing.crop3D import *

#from perso.deformImageAndMask import deformImageAndMaskAndComputeDRRs
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

    print('Deformations and projections finished for deformation', deformation.name)

    return [image, mask]

if __name__ == '__main__':
    ## paths selection ------------------------------------
    
    patientFolder = 'Patient_0'
    dataSetFolder = '/test/'
    dataSetDataFolder = 'data/'
    dataPath = 'E:/Data/liver/' + patientFolder + '/dynModAndROIs.p'
    savingPath = 'E:/Data/liver/' + patientFolder + dataSetFolder

    dataPath = '/export/home/grotsartdehe/dynModAndROIs.p'
    savingPath = '/export/home/grotsartdehe/Data_test/1/'
    if not os.path.exists(savingPath):
        os.makedirs(savingPath)   # Create a new directory because it does not exist
        os.makedirs(savingPath + dataSetDataFolder)  # Create a new directory because it does not exist
        print("New directory created to save the data: ", savingPath)

    # parameters selection ------------------------------------


    sequenceDurationInSecs = 3
    samplingFrequency = 4
    subSequenceSize = 50
    outputSize = [64, 64]
    bodyContourToUse = 'body'
    otherContourToUse = 'MidP CT GTV'
    marginInMM = [50, 0, 100]

    # breathing signal parameters
    amplitude = 'model'
    variationAmplitude = 2
    breathingPeriod = 4
    variationFrequency = 0.1
    shift = 2
    meanNoise = 0
    varianceNoise = 0.5
    samplingPeriod = 1 / samplingFrequency
    simulationTime = sequenceDurationInSecs
    meanEvent = 2 / 30

    # use Z - 0 for Coronal and Z - 90 for sagittal
    projAngle = 0
    projAxis = 'Z'

    multiprocessing = True
    maxMultiProcUse = 12
    tryGPU = True


    ## ------------------------------------------------------------------------------------
    
    ## ------------------------------------------------------------------------------------


    patient = loadDataStructure(dataPath)[0]
    dynMod = patient.getPatientDataOfType("Dynamic3DModel")[0]
    rtStruct = patient.getPatientDataOfType("RTStruct")[0]

    ## get the ROI and mask on which we want to apply the motion signal
    print('Available ROIs')
    rtStruct.print_ROINames()

    gtvContour = rtStruct.getContourByName(otherContourToUse)
    GTVMask = gtvContour.getBinaryMask(origin=dynMod.midp.origin, gridSize=dynMod.midp.gridSize, spacing=dynMod.midp.spacing)
    gtvBox = getBoxAroundROI(GTVMask)

    ## get the body contour to adjust the crop in the direction of the DRR projection
    bodyContour = rtStruct.getContourByName(bodyContourToUse)
    bodyMask = bodyContour.getBinaryMask(origin=dynMod.midp.origin, gridSize=dynMod.midp.gridSize, spacing=dynMod.midp.spacing)
    bodyBox = getBoxAroundROI(bodyMask)

    if projAngle == 0 and projAxis == 'Z': # coronal
        croppingBox = [gtvBox[0], bodyBox[1], gtvBox[2]] ## create the used box combining the two boxes
    elif projAngle == 90 and projAxis == 'Z': # sagittal
        croppingBox = [bodyBox[0], gtvBox[1], gtvBox[2]]
    elif projAngle == 0 and projAxis == 'X': # coronal
        croppingBox = [gtvBox[0], bodyBox[1], gtvBox[2]]
    elif projAngle == 0 and projAxis == 'Y': # sagittal
        croppingBox = [bodyBox[0], gtvBox[1], gtvBox[2]]
    else:
        print('Do not know how to handle crop in this axis/angle configuration, so the body is used')
        croppingBox = [bodyBox[0], bodyBox[1], bodyBox[2]]

    ## crop the model data using the box
    crop3DDataAroundBox(dynMod, croppingBox, marginInMM=marginInMM)

    ## get the mask in cropped version (the dynMod.midp is now cropped so its origin and gridSize has changed)
    GTVMask = gtvContour.getBinaryMask(origin=dynMod.midp.origin, gridSize=dynMod.midp.gridSize, spacing=dynMod.midp.spacing)

    ## if you want to see the crop in the GUI you can save the data in cropped version
    saveSerializedObjects(patient, savingPath + 'croppedModelAndROIs')

    ## get the 3D center of mass of this ROI
    gtvCenterOfMass = gtvContour.getCenterOfMass(dynMod.midp.origin, dynMod.midp.gridSize, dynMod.midp.spacing)
    gtvCenterOfMassInVoxels = getVoxelIndexFromPosition(gtvCenterOfMass, dynMod.midp)
    print('Used ROI name', gtvContour.name)
    print('Used ROI center of mass :', gtvCenterOfMass)
    print('Used ROI center of mass in voxels:', gtvCenterOfMassInVoxels)

    if amplitude == 'model':
        ## to get amplitude from model !!! it takes some time because 10 displacement fields must be computed just for this
        modelValues = getAverageModelValuesAroundPosition(gtvCenterOfMass, dynMod, dimensionUsed='Z')
        amplitude = np.max(modelValues) - np.min(modelValues)
        print('Amplitude of deformation at ROI center of mass', amplitude)

    ## Signal creation
    newSignal = SyntheticBreathingSignal(amplitude=amplitude,
                                         variationAmplitude=variationAmplitude,
                                         breathingPeriod=breathingPeriod,
                                         variationFrequency=variationFrequency,
                                         shift=shift,
                                         meanNoise=meanNoise,
                                         varianceNoise=varianceNoise,
                                         samplingPeriod=samplingPeriod,
                                         simulationTime=sequenceDurationInSecs,
                                         meanEvent=meanEvent)

    #newSignal.generateBreathingSignal()
    newSignal.generate1DBreathingSignal()
    pointList = [gtvCenterOfMass]
    pointVoxelList = [gtvCenterOfMassInVoxels]
    signalList = [newSignal.breathingSignal]

    saveSerializedObjects([signalList, pointList], savingPath + 'ROIsAndSignalObjects')


    ## to show signals and ROIs
    ## -------------------------------------------------------------
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure(figsize=(12, 6))
    signalAx = plt.subplot(2, 1, 2)

    for pointIndex, point in enumerate(pointList):
        ax = plt.subplot(2, len(pointList), pointIndex+1)
        ax.set_title('Slice Y:' + str(pointVoxelList[pointIndex][1]))
        ax.imshow(np.rot90(dynMod.midp.imageArray[:, pointVoxelList[pointIndex][1], :]))
        ax.scatter([pointVoxelList[pointIndex][0]], [dynMod.midp.imageArray.shape[2] - pointVoxelList[pointIndex][2]], c=colors[pointIndex], marker="x", s=100)
        signalAx.plot(newSignal.timestamps/1000, signalList[pointIndex], c=colors[pointIndex])

    signalAx.set_xlabel('Time (s)')
    signalAx.set_ylabel('Deformation amplitude in Z direction (mm)')
    plt.savefig(savingPath + 'ROI_And_Signals_fig.pdf', dpi=300)
    plt.show()

    ## -------------------------------------------------------------

    sequenceSize = newSignal.breathingSignal.shape[0]
    print('Sequence Size =', sequenceSize, 'split by stack of ', subSequenceSize, '. Multiprocessing =', multiprocessing)

    subSequencesIndexes = [subSequenceSize * i for i in range(math.ceil(sequenceSize/subSequenceSize))]
    subSequencesIndexes.append(sequenceSize)
    print('Sub sequences indexes', subSequencesIndexes)

    #startTime = time.time()

    if multiprocessing == False:

        resultList = []

        for i in range(len(subSequencesIndexes)-1):
            print('Creating deformations for images', subSequencesIndexes[i], 'to', subSequencesIndexes[i + 1] - 1)

            deformationList = generateDeformationListFromBreathingSignalsAndModel(dynMod,
                                                                                  signalList,
                                                                                  pointList,
                                                                                  signalIdxUsed=[subSequencesIndexes[i], subSequencesIndexes[i+1]],
                                                                                  dimensionUsed='Z',
                                                                                  outputType=np.float32)

            for deformationIndex, deformation in enumerate(deformationList):
                startTime = time.time()
                resultList.append(deformImageAndMaskAndComputeDRRs(dynMod.midp,
                                                                   GTVMask,
                                                                   deformation,
                                                                   projectionAngle=projAngle,
                                                                   projectionAxis=projAxis,
                                                                   outputSize=outputSize,
                                                                   tryGPU=True))
                endTime = time.time()
                print("time", endTime-startTime)


        savingPath += dataSetDataFolder + f'Patient_0_{sequenceSize}_DRRMasksAndCOM'
        saveSerializedObjects(resultList, savingPath + str(sequenceSize))


    elif multiprocessing == True:
        imageList = []

        if subSequenceSize > maxMultiProcUse:  ## re-adjust the subSequenceSize since this will be done in multi processing
            subSequenceSize = maxMultiProcUse
            print('SubSequenceSize put to', maxMultiProcUse, 'for multiprocessing.')
            print('Sequence Size =', sequenceSize, 'split by stack of ', subSequenceSize, '. Multiprocessing =', multiprocessing)
            subSequencesIndexes = [subSequenceSize * i for i in range(math.ceil(sequenceSize / subSequenceSize))]
            subSequencesIndexes.append(sequenceSize)

        for i in range(len(subSequencesIndexes)-1):
            print('Creating deformations for images', subSequencesIndexes[i], 'to', subSequencesIndexes[i + 1] - 1)

            deformationList = generateDeformationListFromBreathingSignalsAndModel(dynMod,
                                                                                  signalList,
                                                                                  pointList,
                                                                                  signalIdxUsed=[subSequencesIndexes[i], subSequencesIndexes[i+1]],
                                                                                  dimensionUsed='Z',
                                                                                  outputType=np.float32)
            
            resultList = []
            print('Start multi process deformation with', len(deformationList), 'deformations')
            startTime = time.time()
            with concurrent.futures.ProcessPoolExecutor() as executor:

                results = executor.map(deformImageAndMaskAndComputeDRRs, repeat(dynMod.midp), repeat(GTVMask), deformationList, repeat(projAngle), repeat(projAxis), repeat(tryGPU), repeat(outputSize))
                resultList += results
            endTime = time.time()
            print("time", endTime-startTime)
            print('ResultList lenght', len(resultList))
            
            
            
            for imageAndMask in resultList:
                image = imageAndMask[0]
                mask = imageAndMask[1]
                DRR = forwardProjection(image, projAngle, axis=projAxis)
                DRRMask = forwardProjection(mask, projAngle, axis=projAxis)
    
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
    
                # del image  # to release the RAM
                # del mask  # to release the RAM
                
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
                
                imageList.append([croppedDRR, binaryDRRMask, centerOfMass])
            
            endprojTime = time.time()
            print("time", endprojTime-startTime)
            
            

            
        savingPath += dataSetDataFolder + f'Patient_0_{sequenceSize}_DRRMasksAndCOM_multiProcTest'
        saveSerializedObjects(imageList, savingPath)

    stopTime = time.time()
    print('Test with multiprocessing =', multiprocessing, '. Sub-sequence size:', str(subSequenceSize), 'finished in', np.round(stopTime - startTime, 2) / 60, 'minutes')
    print(np.round((stopTime - startTime)/len(resultList), 2), 'sec per sample')
   