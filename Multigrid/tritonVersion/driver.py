import numpy as np
import scipy as sc
import time

import fine2coarse
import coarse2fine
import convergenceTools
import matrixConstructionTools
import solvers
import outputData


dataSize = 193
boundaries = [10,20,15,50,33,75] #in 3d this is 6 values
initialGuess = 0

generateFunction = lambda x,y,z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
trueFunction = lambda x,y,z: -3*np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

originalGrid = np.zeros((dataSize,dataSize,dataSize))
rhs = np.zeros((dataSize,dataSize,dataSize))
trueGrid = np.zeros((dataSize,dataSize,dataSize))

grid = matrixConstructionTools.buildBoundary(originalGrid, boundaries)
#since our initial guess is zero we can leave this as is

xMax = 1
xMin = 0
yMax = 1
yMin = 0
zMax = 1
zMin = 0

deltaXYZ = (xMax-xMin)/(dataSize-1) #since our grid is the same length in all dimensions, we can basically just have one distance in space


rhs = matrixConstructionTools.fillInValues(rhs,generateFunction,deltaXYZ)  #this is wheere we put the function
rhs = matrixConstructionTools.buildBoundary(rhs,[0,0,0,0,0,0])    #since we know the boundaries the rhs should be zero if I recall correctly


for i in range(4):
    grid = solvers.GaussSolve(dataSize,grid,rhs,[(xMax-xMin)/(dataSize-1),(xMax-xMin)/(dataSize-1)])


resGrid = convergenceTools.calcResidual(res, grid,rhs,deltaXYZ,deltaXYZ,deltaXYZ)


#----------------------1st coarsening and solving-----------------------
fineDataSize = dataSize
coarseDataSize = fineDataSize/2 + 1
coarseGridDeltaXYZ = deltaXYZ*2


corseGrid = fine2coarse.bringDown(resGrid)
coarseRHS = fine2coarse.bringDown(rhs)

#we take the un-solved as input and the output is the solved *and* shrunk one)



for i in range(6):
    coarseGrid = solvers.GaussSolve(coarseDataSize,resGrid,rhs,)


#----------------------2nd coarsening and solving-----------------------
fineDataSize = coarseDataSize
coarseDataSize = fineDataSize/2 + 1
coarseGridDeltaXYZ = coarseGridDeltaXYZ*2

corseGrid = fine2coarse.bringDown(coarseGrid)
coarseRHS = fine2coarse.bringDown(rhs)

for i in range(8):
    coarseGrid = solvers.GaussSolve()


#----------------------3rd coarsening and solving-----------------------
fineDataSize = coarseDataSize
coarseDataSize = fineDataSize/2 + 1
coarseGridDeltaXYZ = coarseGridDeltaXYZ*2

corseGrid = fine2coarse(coarseGrid)
coarseRHS = fine2coarse(rhs)

for i in range(12):
    coarseGrid = solvers.GaussSolve()


#----------------------last coarsening and solving-----------------------
fineDataSize = coarseDataSize
coarseDataSize = fineDataSize/2 + 1
coarseGridDeltaXYZ = coarseGridDeltaXYZ*2

corseGrid = fine2coarse.bringDown(coarseGrid)
coarseRHS = fine2coarse.bringDown(rhs)

for i in range(30):
    coarseGrid = solvers.GaussSolve()


#----------------------1st interpolation and solving-----------------------
fineDataSize = coarseDataSize*2-1
fineGridDeltaXYZ = coarseGridDeltaXYZ*2

fineGrid  = coarse2fine.bringUp(coarseGrid)
fineRHS   = coarse2fine.bringUp(rhs)

for i in range(12):
    fineGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = coarse2fine.bringUp()

#----------------------2nd interpolation and solving-----------------------
coarseDataSize = fineDataSize
fineDataSize = coarseDataSize*2-1
fineGridDeltaXYZ = coarseGridDeltaXYZ*2


fineGrid  = coarse2fine.bringUp(fineGrid)
fineRHS   = coarse2fine.bringUp(rhs)

for i in range(8):
    fineGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = convergenceTools.

#----------------------3rd interpolation and solving-----------------------
coarseDataSize = fineDataSize
fineDataSize = coarseDataSize*2-1
fineGridDeltaXYZ = coarseGridDeltaXYZ*2


fineGrid  = coarse2fine.bringUp(fineGrid)
fineRHS   = coarse2fine.bringUp(rhs)

for i in range(6):
    fineGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = convergenceTools.

#------------------back to top and fix---------------------------------------
coarseDataSize = fineDataSize
fineDataSize = coarseDataSize*2-1
fineGridDeltaXYZ = coarseGridDeltaXYZ*2


fineGrid  = coarse2fine.bringUp(fineGrid)
fineRHS   = coarse2fine.bringUp(rhs)

