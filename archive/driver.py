import numpy as np
import scipy as sc
import time

import fine2coarse
import coarse2fine
import convergenceTools
import matrixConstructionTools
import solvers
import outputData


dataSize = 257
boundaries = [10,20,15,50,33,75] #in 3d this is 6 values (6 faces)
initialGuess = 0

generateFunction = lambda x,y,z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
trueFunction = lambda x,y,z: -3*np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
originalGrid = np.zeros((dataSize,dataSize,dataSize))
rhs = np.zeros((dataSize,dataSize,dataSize))
trueGrid = np.zeros((dataSize,dataSize,dataSize))

grid = matrixConstructionTools.buildBoundary(originalGrid, boundaries)
print(grid.shape)
#since our initial guess is zero we can leave this as is

rhs = matrixConstructionTools.fillInValues(rhs,generateFunction)  #this is wheere we put the function
rhs = matrixConstructionTools.buildBoundary(rhs,[0,0,0,0,0,0])    #since we know the boundaries the rhs should be zero if I recall correctly

for i in range(4):
    grid = solvers.GaussSolve(grid,)


currentGrid = convergenceTools.calcResidual(grid,)


#----------------------1st coarsening and solving-----------------------
fineDataSize = dataSize
coarseDataSize = fineDataSize/2 + 1

corseGrid = fine2coarse(currentGrid)
coarseRHS = fine2coarse(rhs)

#we take the un-solved as input and the output is the solved *and* shrunk one)



for i in range(6):
    coarseGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = convergenceTools.

#----------------------2nd coarsening and solving-----------------------
fineDataSize = coarseDataSize
coarseDataSize = fineDataSize/2 + 1

corseGrid = fine2coarse()
coarseRHS = fine2coarse()

for i in range(8):
    coarseGrid = solvers.GaussSolve()


#----------------------3rd coarsening and solving-----------------------
fineDataSize = coarseDataSize
coarseDataSize = fineDataSize/2 + 1

corseGrid = fine2coarse()
coarseRHS = fine2coarse()

for i in range(12):
    coarseGrid = solvers.GaussSolve()


#----------------------last coarsening and solving-----------------------
fineDataSize = coarseDataSize
coarseDataSize = fineDataSize/2 + 1

corseGrid = fine2coarse()
coarseRHS = fine2coarse()

for i in range(30):
    coarseGrid = solvers.GaussSolve()


#----------------------1st interpolation and solving-----------------------
fineDataSize = coarseDataSize*2-1

fineGrid  = coarse2fine.bringUp()
fineRHS   = coarse2fine.bringUp()

for i in range(12):
    fineGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = coarse2fine.bringUp()

#----------------------2nd interpolation and solving-----------------------
coarseDataSize = fineDataSize
fineDataSize = coarseDataSize*2-1

fineGrid  = coarse2fine.
fineRHS   = coarse2fine.

for i in range(8):
    fineGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = convergenceTools.

#----------------------3rd interpolation and solving-----------------------
coarseDataSize = fineDataSize
fineDataSize = coarseDataSize*2-1

fineGrid  = coarse2fine.
fineRHS   = coarse2fine.

for i in range(6):
    fineGrid = solvers.GaussSolve()

currentGrid = convergenceTools.
currentRHS = convergenceTools.

#------------------back to top and fix---------------------------------------
coarseDataSize = fineDataSize
fineDataSize = coarseDataSize*2-1

fineGrid  = coarse2fine.
fineRHS   = coarse2fine.
fineError = coarse2fine.

