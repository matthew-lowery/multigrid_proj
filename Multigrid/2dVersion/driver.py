import numpy as np
import scipy as sc
import time

import fine2coarse
import coarse2fine
import convergenceTools
import matrixConstructionTools
import solvers
import outputData


#Fuck it, lets stick with what we know works in the C code
gridSize = 129



dataSize = gridSize
boundaries = [0,0,0,0] #in 3d this is 6 values
initialGuess = 0

originalGrid = np.zeros((dataSize,dataSize))
rhs = np.zeros((dataSize,dataSize))
trueGrid = np.zeros((dataSize,dataSize))

generateFunction = lambda x,y: np.sin(np.pi * x) * np.sin(np.pi * y)
trueFunction = lambda x,y: -*np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

print("Creating guess matrix")
grid = matrixConstructionTools.buildBoundary(originalGrid, boundaries)
#since our initial guess is zero we can leave this as is
print("Done")

print("Current grid size", dataSize)


xMax = 1
xMin = 0
yMax = 1
yMin = 0

deltaX = (xMax-xMin)/(dataSize-1) #since our grid is the same length in all dimensions, we can basically just have one distance in space
deltaY = (yMax-yMin)/(dataSize-1)

print("Constructing RHS matrix")
rhs = matrixConstructionTools.fillInValues(rhs,generateFunction,[deltaX,deltaY])  #this is wheere we put the function
rhs = matrixConstructionTools.buildBoundary(rhs,[0,0,0,0])    #since we know the boundaries the rhs should be zero if I recall correctly

print("Done")
print("Smoothing current grid")
for i in range(4):
    grid = solvers.GaussSolve(dataSize,grid,rhs,[deltaX,deltaY])
print("Done")

print("Calculating residual")
resGrid = np.zeros_like(grid)
resGrid = convergenceTools.calcResidual(dataSize, resGrid, grid,rhs,[deltaX,deltaY])
print("Done")

#----------------------1st coarsening and solving-----------------------
print("Calculating 1st coarse grid size")

fineSize = gridSize
coarseSize = int(fineSize/2 + 1)

fineDeltaX = deltaX
fineDeltaY = deltaY

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2

print("Done, new grid is ",coarseSize,"x",coarseSize)

#WE CHECK THE DIVISION IS THE SAME


print("Coarsening grid and RHS")
coarseGrid = fine2coarse.bringDown(resGrid,coarseSize,fineSize)
coarseRHS = fine2coarse.bringDown(rhs,coarseSize,fineSize)
print("Done")

#we take the un-solved as input and the output is the solved *and* shrunk one)


print("Smoothing current grid")
for i in range(6):
    coarseGrid = solvers.GaussSolve(coarseSize,coarseGrid,coarseRHS,[coarseDeltaX,coarseDeltaY])
print("Done")


#----------------------2nd coarsening and solving-----------------------
print("Calculating 2nd coarse grid size")

fineSize = coarseSize
coarseSize = int(fineSize/2 + 1)

fineDeltaX = coarseDeltaX
fineDeltaY = coarseDeltaY

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2


print("Done, new grid is ",coarseSize,"x",coarseSize)

print("Coarsening grid and RHS")
coarseGrid = fine2coarse.bringDown(resGrid,coarseSize,fineSize)
coarseRHS = fine2coarse.bringDown(coarseRHS,coarseSize,fineSize)
print("Done")

print("Smoothing current grid")
for i in range(8):
    coarseGrid = solvers.GaussSolve(coarseSize,coarseGrid,coarseRHS,[coarseDeltaX,coarseDeltaY])
print("Done")




#----------------------last coarsening and solving-----------------------
print("Calculating 3rd coarse grid size")
fineSize = coarseSize
coarseSize = int(fineSize/2 + 1)

fineDeltaX = coarseDeltaX
fineDeltaY = coarseDeltaY
fineDeltaZ = coarseDeltaZ

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2

print("Done, new grid is ",coarseSize,"x",coarseSize)

print("Coarsening grid and RHS")
coarseGrid = fine2coarse.bringDown(resGrid,coarseSize,fineSize)
coarseRHS = fine2coarse.bringDown(coarseRHS,coarseSize,fineSize)
print("Done")

print("Smoothing current grid")
for i in range(50):
    coarseGrid = solvers.GaussSolve(coarseSize,coarseGrid,coarseRHS,[coarseDeltaX,coarseDeltaY,coarseDeltaZ])
print("Done")

print("DEBUG: COMPARE OUTPUT HERE TO C VERSION")
print(coarseGrid[1])


#----------------------1st interpolation and solving-----------------------
print("Calculating 1st refined grid size")
fineSize = coarseSize * 2 -1

fineDeltaX = coarseDeltaX/2
fineDeltaY = coarseDeltaY/2
fineDeltaZ = coarseDeltaZ/2

print("Done, new grid is ",fineSize,"x",fineSize,"x",fineSize)

print("Refining and interpolating grid")
fineGrid  = coarse2fine.bringUp(coarseGrid,fineSize,coarseSize)
fineRHS   = coarse2fine.bringUp(coarseRHS,fineSize,coarseSize)
print("Done")


print("Smoothing current grid")
for i in range(12):
    fineGrid = solvers.GaussSolve(fineSize,fineGrid,fineRHS,[fineDeltaX,fineDeltaY,fineDeltaZ])
print("Done")


#----------------------2nd interpolation and solving-----------------------
print("Calculating 2nd refined grid size")
coarseSize = fineSize
fineSize = coarseSize * 2 -1

coarseDeltaX = fineDeltaX
coarseDeltaY = fineDeltaY
coarseDeltaZ = fineDeltaZ

fineDeltaX = coarseDeltaX/2
fineDeltaY = coarseDeltaY/2
fineDeltaZ = coarseDeltaZ/2

print("Done, new grid is ",fineSize,"x",fineSize,"x",fineSize)

print("Refining and interpolating grid")
fineGrid  = coarse2fine.bringUp(fineGrid,fineSize,coarseSize)
fineRHS   = coarse2fine.bringUp(fineRHS,fineSize,coarseSize)
print("Done")

for i in range(8):
    fineGrid = solvers.GaussSolve(fineSize,fineGrid,fineRHS,[fineDeltaX,fineDeltaY,fineDeltaZ])
print("Done")

#----------------------3rd interpolation and solving-----------------------
print("Calculating 3rd refined grid size")
coarseSize = fineSize
fineSize = coarseSize * 2 -1

coarseDeltaX = fineDeltaX
coarseDeltaY = fineDeltaY
coarseDeltaZ = fineDeltaZ

fineDeltaX = coarseDeltaX/2
fineDeltaY = coarseDeltaY/2
fineDeltaZ = coarseDeltaZ/2

print("Done, new grid is ",fineSize,"x",fineSize,"x",fineSize)

print("Refining and interpolating grid")
fineGrid  = coarse2fine.bringUp(fineGrid,fineSize,coarseSize)
fineRHS   = coarse2fine.bringUp(fineRHS,fineSize,coarseSize)
print("Done")

for i in range(8):
    fineGrid = solvers.GaussSolve(fineSize,fineGrid,fineRHS,[fineDeltaX,fineDeltaY,fineDeltaZ])
print("Done")

output = originalGrid - fineGrid
trueVal = np.zeros_like(originalGrid)
trueVal = matrixConstructionTools.fillInValues(trueVal,trueFunction,[deltaX,deltaY,deltaZ])




trueVal = matrixConstructionTools.buildBoundary(trueVal, boundaries)
print(trueVal - output)

