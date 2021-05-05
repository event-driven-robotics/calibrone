# -*- coding: utf-8 -*-
"""
Code contributions from Sim Bamford and Suman Ghosh

This script steps through converting an event recording to a video,
followed by single-camera calibration based on a checkboard pattern.

This script is intended to be run in "scientific mode"
(execution one cell at a time, from an IDE)

It requires:
    (a) https://github.com/event-driven-robotics/rpg_e2vid
    - note that this has its own dependencies - follow the installation
    instructions in that repo for best results.
    (b) opencv (pip install opencv-python)

"""

#%% Set up folders

import os, sys
prefix = 'C:/' if os.name == 'nt' else '/home/sbamford/' # YMMV: the root for your repos
sys.path.insert(0, os.path.join(prefix, 'repos')) # path to rpg_e2vid
sys.path.insert(0, os.path.join(prefix, 'repos/bimvee'))
sys.path.insert(0, os.path.join(prefix, 'repos/mustard')) # If you want to visualise data

#%% Run reconstruction

from rpg_e2vid.run_reconstruction import run_reconstruction

filePathRoot = os.path.join(prefix, 
                              'data',
                              '2020_09_30_Aiko_KitchenVicon',
                              '2020-09-30_calib01',
                              'StEFI')

filePathOrName=os.path.join(filePathRoot, 'StEFI')
exportFilePathOrName = os.path.join(filePathRoot,'frames')

kwargs = {}
kwargs['input_file'] = filePathOrName
kwargs['path_to_model'] = os.path.join(prefix, 'repos/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar')
kwargs['output_folder'] = exportFilePathOrName
kwargs['auto_hdr'] = True
kwargs['display'] = True
kwargs['show_events'] = True
kwargs['channelName'] = 'right'

run_reconstruction(**kwargs)
pathToFrames = os.path.join(kwargs['output_folder'], 'reconstruction')

#%% Now take those frames and put them in the right place ...

#%% Import frames

from bimvee.importAe import importAe

containerFrames = importAe(filePathOrName=pathToFrames)

#%% You may like to check that the container looks correct

from bimvee.info import info

info(containerFrames)

#%%  You may like to visualise the container - Start Mustard

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

app.setData(containerFrames)

#%% Unpack the frames from the container

from bimvee.container import Container
containerObj = Container(containerFrames)
frames = containerObj.getDataType('frame')

#%% Optionally, downsample the frames

import numpy as np
from bimvee.split import getSamplesAtTimes

numSamples = 100
firstTime = frames['ts'][0]
lastTime = frames['ts'][-1]
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

