# -*- coding: utf-8 -*-
"""
Code contributions from Sim Bamford and Suman Ghosh

This script steps through:
    
    Importing data from event camera and vicon
    Conversion of event data to frames (using e2vid network)
    Single camera calibration
    Hand-eye calibration

"""

#%% Preliminaries

import os, sys
prefix = 'C:/' if os.name == 'nt' else '/home/sbamford/'
sys.path.insert(0, os.path.join(prefix, 'repos'))
sys.path.insert(0, os.path.join(prefix, 'repos/bimvee'))
sys.path.insert(0, os.path.join(prefix, 'repos/mustard'))

#%% Load event data

from bimvee.importAe import importAe

filePathOrName = os.path.join(prefix, 
                              'data',
                              '2020_09_30_Aiko_KitchenVicon',
                              '2020-09-30_calib01')

filePathOrName = os.path.join(prefix, 
                              'data',
                              '2020_10_08_AikoSim_KitchenVicon',
                              '2020-10-08_calib00')

containerEvents = importAe(filePathOrName=os.path.join(filePathOrName, 'StEFI'))

#%%

from bimvee.info import info

info(containerEvents)

#%% Run reconstruction

from rpg_e2vid.run_reconstruction import run_reconstruction

exportFilePathOrName = os.path.join(filePathOrName,'frames')

kwargs = {}
kwargs['input_file'] = filePathOrName=os.path.join(filePathOrName, 'StEFI')
kwargs['path_to_model'] = os.path.join(prefix, 'repos/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar')
kwargs['output_folder'] = exportFilePathOrName
kwargs['auto_hdr'] = True
kwargs['display'] = True
kwargs['show_events'] = True
kwargs['channelName'] = 'right'

run_reconstruction(**kwargs)
pathToFrames = os.path.join(kwargs['output_folder'], 'reconstruction')

#%% Take those frames and put them in the right place ...

#%% Import frames

from bimvee.importAe import importAe

containerFrames = importAe(filePathOrName=os.path.join(filePathOrName, 'frames'), 
                           zeroTime=False)

#%%

from bimvee.info import info

info(containerFrames)

#%% Import poses

import numpy as np
from bimvee.importAe import importAe
from bimvee.split import selectByBool

containerVicon = importAe(filePathOrName=os.path.join(filePathOrName, 'Vicon'))
# Filter out just the desired pose samples
from bimvee.importIitVicon import separateMarkersFromSegments
containerVicon['data']['vicon'] = separateMarkersFromSegments(containerVicon['data']['vicon']['pose6q'])
del containerVicon['data']['vicon']['point3']

# Remove null pose samples (where tracking failed)
nullPoses = containerVicon['data']['vicon']['pose6q']['point'] == np.zeros((1, 3))
nullPoses = np.any(nullPoses, axis = 1)

containerVicon['data']['vicon']['pose6q'] = selectByBool(
    containerVicon['data']['vicon']['pose6q'], ~nullPoses)


#%%

from bimvee.info import info

info(containerVicon)

#%% Time alignment

'''
Now for some irreducible complexity
events and vicon were simultaneuous yarp imports. They need to be time-aligned 
using tsOffsetFromInfo.
Reconstructed frames takes the same timestamp as the dvs events - they need 
to be aligned according to those. Here we just use the coincidence that dvs 
recording started before vicon recording, so dvs events happen to be aligned 
to more or less zero
'''

from bimvee.timestamps import rezeroTimestampsForImportedDicts
containers = [containerEvents, containerVicon]
rezeroTimestampsForImportedDicts(containers)

#%% Combine containers

container = containerVicon 
container['data'].update(containerFrames['data'])
# Now that we've used event data for time alignment, we don't need it any more,
# so we don't merge in containerEvents

#%% Check that the container looks correct
from bimvee.info import info
info(container)

#%% Visualise the container - Start Mustard

cwd = os.getcwd() 

import threading
import mustard
app = mustard.Mustard()
thread = threading.Thread(target=app.run)
thread.daemon = True
thread.start()

#%% Once mustard is open, undo the change of working directory

os.chdir(cwd)

#%% Visualise

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container

#%% Export - import

# At this point we could export the container for future use. 
# Here I reload data which has been exported as a rosbag

from bimvee.importAe import importAe

filePathOrName = os.path.join(prefix,
                              'data',
                              '2020_06_26_Aiko_Checkerboard',
                              'trial08ReconBag',
                              'checkerboard.bag')

container = importAe(filePathOrName=filePathOrName)

#%% Unpack

from bimvee.container import Container
containerObj = Container(container)

frames = containerObj.getDataType('frame')
poses = containerObj.getDataType('pose6q')

#%% Optionally, downsample the frames

import numpy as np
from bimvee.split import getSamplesAtTimes

numSamples = 300
firstTime = poses['ts'][0]
lastTime = poses['ts'][-1]
frames = getSamplesAtTimes(frames, 
                           np.arange(firstTime, lastTime, 
                                     (lastTime - firstTime) / numSamples),
                           allowDuplicates = False)

#%% Find checkerboard patterns in the frames

import numpy as np
import cv2
from bimvee.split import getSamplesAtTimes, selectByBool
from tqdm import tqdm

# Checkerboard (7x4 vertices, 35mm)
squareEdgeLength = 0.0383  # m
checkerboardDims = (5, 4)
meshgrid = np.meshgrid(np.arange(checkerboardDims[0]) * squareEdgeLength, 
                       np.arange(checkerboardDims[1]) * squareEdgeLength)
x = meshgrid[0].flatten()
y = meshgrid[1].flatten()
z = np.zeros_like(y)

# generate calibration pattern coords
coords = np.concatenate((x[:, np.newaxis], 
                         y[:, np.newaxis], 
                         z[:, np.newaxis]), 
                        axis=1).astype(np.float32)
usableImagePoints = []
usableBool = np.zeros((len(frames['frames'])), dtype=np.bool)
for idx, frame in enumerate(tqdm(frames['frames'], file=sys.stdout)):
    corners = cv2.findChessboardCorners(frames['frames'][idx], 
                                        patternSize=checkerboardDims)
    if corners[0] == True:
        usableBool[idx] = True
        usableImagePoints.append(corners[1][:, 0, :])
frames = selectByBool(frames, usableBool)
frames['imagePoints'] = usableImagePoints

#%% Select poses which match the frames

poses = getSamplesAtTimes(poses, frames['ts'], allowDuplicates=True)

#%% Filter poses too far from frames

import math

maxTimeDifference = 0.01

numSamples = poses['ts'].shape[0]
keepBool = np.ones((numSamples), dtype=np.bool)
for idx in range(numSamples):
    timeDifference = math.fabs(poses['ts'][idx] - frames['ts'][idx])
    if timeDifference > maxTimeDifference:
        keepBool[idx] = False

from bimvee.split import selectByBool

frames = selectByBool(frames, keepBool)
poses = selectByBool(poses, keepBool)

#%% Combine the frames and poses dicts

# (this discards one of the timestamp arrays, but now we consider them sufficiently similar) 
frames.update(poses)

#%% Another optional point at which to downsample the frames

framesBeforeCalibration = frames

from bimvee.split import getSamplesAtTimes

numSamples = 100
lastTime = frames['ts'][-1]
frames = getSamplesAtTimes(frames, 
                           np.arange(0, lastTime, lastTime / numSamples),
                           allowDuplicates = False)

#%% Run the calibration

#frames = framesBeforeCalibration 

objectPoints = [coords for idx in range(len(frames['frames']))]
# camera matrix is an initial guess
cameraMatrix = np.array([[200, 0, 152],
                         [0, 200, 120],
                         [0, 0, 1]], dtype=np.float64)
distCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)
retval, cameraMatrix, distCoeffs, rVecs, tVecs = cv2.calibrateCamera(
    objectPoints=objectPoints, 
    imagePoints=frames['imagePoints'], 
    imageSize=(240, 304),
    cameraMatrix=cameraMatrix,
    distCoeffs=distCoeffs)

# store the resulting rvecs and tvecs in the dict

frames['rVecs'] = rVecs
frames['tVecs'] = tVecs
        
print('reprojectionError = ' + str(retval))
print('cameraMatrix = ')
print(cameraMatrix)
print('distortionCoefficients = ')
print(distCoeffs)

#%% Calculate the reprojection errors by frame

import matplotlib.pyplot as plt
        
numCoords = coords.shape[0]
reprojectionErrors = []
for frameIdx in range(len(frames['frames'])):
    reprojectedImagePoints, _ = cv2.projectPoints(objectPoints=coords,
                                   rvec=frames['rVecs'][frameIdx],
                                   tvec=frames['tVecs'][frameIdx],
                                   cameraMatrix=cameraMatrix,
                                   distCoeffs=distCoeffs)
    # remove singleton dimension
    reprojectedImagePoints = reprojectedImagePoints[:, 0, :]
    imagePoints = frames['imagePoints'][frameIdx]
    pointsDiff = reprojectedImagePoints - imagePoints
    reprojectionErrors.append(np.mean(np.linalg.norm(pointsDiff, axis=1)))
    
frames['reprojectionErrors'] = np.array(reprojectionErrors)
reprojectionErrorsSorted = reprojectionErrors.copy()
reprojectionErrorsSorted.sort()
plt.figure()
plt.plot(reprojectionErrorsSorted, 'o')
plt.xlabel('frame number, sorted by ascending reprojection error')
plt.ylabel('Mean reprojection error (pixels)')

#%% Select just the frames with the lowest reprojection errors

framesBeforeRejectingHighReprojectionErrors = frames
# frames = framesBeforeRejectingHighReprojectionErrors

# Use the above result to decide where the threshold should be
reprojectionErrorThreshold = 0.2

keepBool = frames['reprojectionErrors'] < reprojectionErrorThreshold 
frames = selectByBool(frames, keepBool)
framesSelectedForHandEyeCalibration = frames

#%% Hand-eye calibration

'''
A is pose from vicon to dvs trackable
B is transform from dvs camera viewpoint to checkerboard
'''

from bimvee.geometry import quat2RotM
from calibrone.calibrationFunctions import pose_estimation, invertTransformationMatrices

frames = framesSelectedForHandEyeCalibration 

numFrames = len(frames['frames'])

# First, the vicon to dvs-trackable poses
aViconToStefiRotMats = np.zeros((4, 4, numFrames), np.float64)

for idx in range(numFrames):
    aViconToStefiRotMats[:, :, idx] = quat2RotM(frames['rotation'][idx])
    aViconToStefiRotMats[:3, 3, idx] = frames['point'][idx]

# Secondly, the dvs-camera-viewpoint to checkerboard poses
bAtisToCheckerboardRotMats = np.zeros((4, 4, numFrames), np.float64)
bAtisToCheckerboardRotMats[3, 3, :] = 1
for idx in range(numFrames):
    bAtisToCheckerboardRotMats[:3, :3, idx] = cv2.Rodrigues(frames['rVecs'][idx])[0]
    bAtisToCheckerboardRotMats[:3, 3, idx] = frames['tVecs'][idx][:, 0]

a = aViconToStefiRotMats
b = bAtisToCheckerboardRotMats


aInv = invertTransformationMatrices(a)
bInv = invertTransformationMatrices(b)

x, y, yCheck, errorStats = pose_estimation(a, bInv)

xFinal = x
yFinal = y

#%% Hand-eye calibration - opencv function - alternative to the above
#
import cv2
#
# following: http://www.graphics.stanford.edu/courses/cs248-98-fall/Final/q4.html
#
'''
@param[in] R_gripper2base Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the robot base frame (\f$_{}^{b}\textrm{T}_g\f$).
This is a vector (`vector<Mat>`) that contains the rotation matrices for all the transformations
from gripper frame to robot base frame.

The robot base frame is the world, or the coord system for the vicon
The gripper frame is dvs-trackable pose
So we want a transformation from dvs-trackable to the world
'''

a = aViconToStefiRotMats
aInv = invertTransformationMatrices(a)
r_gripper2base = a[:3, :3, :]
t_gripper2base = a[:3, 3, :]

'''
@param[in] R_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the target frame to the camera frame (\f$_{}^{c}\textrm{T}_t\f$).
This is a vector (`vector<Mat>`) that contains the rotation matrices for all the transformations
from calibration target frame to camera frame.

These are the rvecs and tvecs, but I don't know if I have them around the right way
... let's try to turn them around
'''
b = bAtisToCheckerboardRotMats
bInv = invertTransformationMatrices(b)
r_target2cam = b[:3, :3, :]
t_target2cam = b[:3, 3, :]

methods = [cv2.CALIB_HAND_EYE_TSAI,
           cv2.CALIB_HAND_EYE_PARK,
           cv2.CALIB_HAND_EYE_HORAUD,
           cv2.CALIB_HAND_EYE_ANDREFF,
           cv2.CALIB_HAND_EYE_DANIILIDIS]

xx = cv2.calibrateHandEye(np.moveaxis(r_gripper2base, 2, 0),
                          np.moveaxis(t_gripper2base, 1, 0),
                          np.moveaxis(r_target2cam, 2, 0),
                          np.moveaxis(t_target2cam, 1, 0),
                          None,
                          None,
                          method=methods[0])
xR = xx[0]
xT = xx[1]
x = np.zeros((4, 4))
x[:3, :3] = xR
x[:3, 3] = xT[:, 0]
x[3, 3] = 1

# Compute vicon-checkerboard transform using estimated X and one of the (a,b) pairs
y = aViconToStefiRotMats[:, :, 0] @ x @ bAtisToCheckerboardRotMats[:, :, 0]

xFinal = x
yFinal = y

#%% Optionally, apply a second stage of optimisation

'''
In the global workspace are coord (the map of checkerboard points)
a and b - the transformations for each frame
x and y come in as the initial guesses that get refined
For conveninece, we calculate the target reprojections in the global workspace
The residual function takes coeffs x and y in rvec/tvec form, so
    [rvecx, tvecx, rvecy, tvecy] - 12 params in a line
It returns an average reprojection error for each frame (Could also treat all reproject errors separately for each corner and each frame,- might work better but might be overkill)
'''

import numpy as np
import scipy.optimize
import cv2
import math

from calibrone.calibrationFunctions import transformationMatrixFromVecs

k = cameraMatrix
d = distCoeffs

ripAll = []
for frameIdx in range(numFrames):
    # points reprojected via board to atis transformation output from single camera calibration
    reprojectedImagePoints, _ = cv2.projectPoints(objectPoints=coords,
                                                  rvec=rVecs[frameIdx],
                                                  tvec=tVecs[frameIdx],
                                                  cameraMatrix=k,
                                                  distCoeffs=d)
    reprojectedImagePoints = reprojectedImagePoints[:, 0, :]
    ripAll.append(reprojectedImagePoints)
rip = np.array(np.concatenate(ripAll), dtype=np.float64)


def residualFunc(coeff):
    ripAll = []
    for frameIdx in range(numFrames):
        # construct x and y from the coeffs
        rVecX = np.array(coeff[0:3])
        tVecX = np.array(coeff[3:6])
        rVecY = np.array(coeff[6:9])
        tVecY = np.array(coeff[9:12])
        x = transformationMatrixFromVecs(rVecX, tVecX)
        y = transformationMatrixFromVecs(rVecY, tVecY)
        yInv = np.linalg.inv(y)
        bForFrame = np.linalg.inv(yInv @ a[:, :, frameIdx] @ x)
        rVecB, _ = cv2.Rodrigues(bForFrame[:3, :3])
        tVecB = bForFrame[:3, 3]
        reprojectedImagePoints, _ = cv2.projectPoints(objectPoints=coords,
                                                      rvec=rVecB,
                                                      tvec=tVecB,
                                                      cameraMatrix=k,
                                                      distCoeffs=d)
        ripAll.append(reprojectedImagePoints[:, 0, :])
        # print(len(ripAll))
    rip2 = np.concatenate(ripAll)
    rip2 = np.array(rip2, dtype=np.float64)
    # we now have rip (reprojected image points) for both routes - calculate Euclidean distances
    res = np.sum(np.sum((rip - rip2) ** 2))
    #print(res)
    return res


rVecX, _ = cv2.Rodrigues(x[:3, :3])
tVecX = x[:3, 3]
rVecY, _ = cv2.Rodrigues(y[:3, :3])
tVecY = y[:3, 3]

# TODO optimize Euler angles instead of rotation vector
init_guess = np.concatenate([rVecX[:, 0], tVecX, rVecY[:, 0], tVecY])
optResult = scipy.optimize.least_squares(residualFunc,
                                         init_guess)
# method='lm')
coeffOpt = optResult['x']

# reinflate result to matrices

rVecXFinal = np.array(coeffOpt[0:3])
tVecXFinal = np.array(coeffOpt[3:6])
rVecYFinal = np.array(coeffOpt[6:9])
tVecYFinal = np.array(coeffOpt[9:12])

xFinal = transformationMatrixFromVecs(rVecXFinal, tVecXFinal)
yFinal = transformationMatrixFromVecs(rVecYFinal, tVecYFinal)

#%% Reproject checkerboard points to frame, including final result

frames = framesBeforeCalibration
numFrames = len(frames['frames'])

k = cameraMatrix
d = distCoeffs
import matplotlib.pyplot as plt

for idx in range(0, numFrames, 10):
    exampleFrame = frames['frames'][idx]
    exampleImagePoints = frames['imagePoints'][idx]

    plt.figure()

    # plot greyscale frame and detected (using OpenCV) corner points
    plt.imshow(exampleFrame, cmap='gray')
    plt.plot(exampleImagePoints[:, 0], exampleImagePoints[:, 1], 'og')

    exampleA = np.zeros((4, 4), np.float64)
    exampleA[:, :] = quat2RotM(frames['rotation'][idx])
    exampleA[:3, 3] = frames['point'][idx]

    #plot reprojected corner points in using refined X and Y poses
    yFinalInv = np.linalg.inv(yFinal)
    exampleBComputed = np.linalg.inv(yFinalInv @ exampleA @ xFinal)
    rVec, _ = cv2.Rodrigues(exampleBComputed[:3, :3])
    tVec = exampleBComputed[:3, 3]
    reprojectedImagePoints, _ = cv2.projectPoints(objectPoints=coords,
                                                  rvec=rVec,
                                                  tvec=tVec,
                                                  cameraMatrix=k,
                                                  distCoeffs=d)
    reprojectedImagePoints = reprojectedImagePoints[:, 0, :]

    plt.plot(reprojectedImagePoints[:, 0], reprojectedImagePoints[:, 1], 'or')

#%%
    
print('x = ')
print(xFinal)    
#%%
from calibrone.calibrationFunctions import invertTransformationMatrix

xInv = invertTransformationMatrix(xFinal)

print('xInv = ')
print(xInv)    


#%% Visualise hand-eye transformed poses

'''
Here it's better to revert to the original frame and pose sequence before 
downsampling
'''

x = np.array([[-0.01051956,  0.99969537,  0.02232708,  0.04740807],
              [ 0.99837317,  0.01175173, -0.05579353,  0.08749821],
              [-0.05603892,  0.02170383, -0.99819266, -0.13020822],
              [ 0.        ,  0.        ,  0.        ,  1.        ]])
xInv = np.array([[-0.01051956,  0.99837317, -0.05603892, -0.09415388],
                 [ 0.99969537,  0.01175173,  0.02170383, -0.04559587],
                 [ 0.02232708, -0.05579353, -0.99819266, -0.12614954],
                 [ 0.        ,  0.        ,  0.        ,  1.        ]])

from bimvee.pose import transform

posesTransformed = transform(poses, transformationMatrix=x, direction='reverse')
posesTransformedInv = transform(poses, transformationMatrix=xInv, direction='reverse')

container['data']['poseTransformed'] = {'pose6q': posesTransformed}
container['data']['poseTransformedInv'] = {'pose6q': posesTransformedInv}

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container


np.linalg.norm(np.array([0.047, 0.087, 0.13]))

