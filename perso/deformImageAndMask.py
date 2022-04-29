import numpy as np

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
