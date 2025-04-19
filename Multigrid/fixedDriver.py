import numpy as np
import scipy as sc
import time

import fine2coarse
import coarse2fine
import convergenceTools
import matrixConstructionTools
import solvers
import outputData

import matplotlib.pyplot as mpl

#Fuck it, lets stick with what we know works in the C code
gridSize = 129



dataSize = gridSize
boundaries = [0,0,0,0,0,0] #in 3d this is 6 values
initialGuess = 0

originalGrid = np.zeros((dataSize,dataSize,dataSize))
rhs = np.zeros((dataSize,dataSize,dataSize))
trueGrid = np.zeros((dataSize,dataSize,dataSize))

generateFunction = lambda x,y,z: -3*np.pi**2 * (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
trueFunction = lambda x,y,z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

print("Creating guess matrix")
grid = matrixConstructionTools.buildBoundary(originalGrid, boundaries)
#since our initial guess is zero we can leave this as is
print("Done")

print("Current grid size", dataSize)


                #4 steps down, 4 steps up

xMax = 1
xMin = 0
yMax = 1
yMin = 0
zMax = 1
zMin = 0

deltaX = (xMax-xMin)/(dataSize-1) #since our grid is the same length in all dimensions, we can basically just have one distance in space
deltaY = (yMax-yMin)/(dataSize-1)
deltaZ = (zMax-zMin)/(dataSize-1)


print("Constructing RHS matrix")
rhs = matrixConstructionTools.fillInValues(rhs,generateFunction,[deltaX,deltaY,deltaZ])  #this is where we put the function
rhs = matrixConstructionTools.buildBoundary(rhs,[0,0,0,0,0,0])    #since we know the boundaries the rhs should be zero if I recall correctly

print("Done")
print("Smoothing current grid")
for i in range(2):
    grid = solvers.GaussSolve(dataSize,grid,rhs,[deltaX,deltaY,deltaZ])
print("Done")

midpoint = int((dataSize-1)/2)

print("mid of top grid", grid[midpoint,midpoint,midpoint] )

print("Calculating residual")
resGrid = np.zeros_like(grid)
resGrid = convergenceTools.calcResidual(dataSize,resGrid, grid,rhs,[deltaX,deltaY,deltaZ])
print("Done")

#----------------------1st coarsening and solving-----------------------
print("Calculating 1st coarse grid size")

fineSize = gridSize
coarseSize = int(fineSize/2 + 1)

fineDeltaX = deltaX
fineDeltaY = deltaY
fineDeltaZ = deltaZ

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2
coarseDeltaZ = fineDeltaZ*2

print("Done, new grid is ",coarseSize,"x",coarseSize,"x",coarseSize)


print("Coarsening grid and RHS")
coarseRes = fine2coarse.bringDown(resGrid,coarseSize,fineSize)
coarseError = np.zeros_like(coarseRes)
print("Done")




#we take the un-solved as input and the output is the solved *and* shrunk one)


print("Smoothing current grid")
for i in range(2):
    coarseError = solvers.GaussSolve(coarseSize,coarseError,coarseRes,[coarseDeltaX,coarseDeltaY,coarseDeltaZ])
print("Done")



#----------------------2nd coarsening and solving-----------------------
print("Calculating 2nd coarse grid size")

fineSize = coarseSize
coarseSize = int(fineSize + 1)/2

fineDeltaX = coarseDeltaX
fineDeltaY = coarseDeltaY
fineDeltaZ = coarseDeltaZ

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2
coarseDeltaZ = fineDeltaZ*2

print("Done, new grid is ",coarseSize,"x",coarseSize,"x",coarseSize)

print("Coarsening grid and RHS")
coarseRes = fine2coarse.bringDown(coarseRes,coarseSize,fineSize)
coarseError = fine2coarse.bringDown(coarseError,coarseSize,fineSize)
print("Done")


print("Smoothing current grid")
for i in range(2):
    coarseError = solvers.GaussSolve(coarseSize,coarseError,coarseRes,[coarseDeltaX,coarseDeltaY,coarseDeltaZ])
print("Done")




#----------------------3nd coarsening and solving-----------------------
print("Calculating 3rd coarse grid size")

fineSize = coarseSize
coarseSize = int(fineSize/2 + 1)

fineDeltaX = coarseDeltaX
fineDeltaY = coarseDeltaY
fineDeltaZ = coarseDeltaZ

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2
coarseDeltaZ = fineDeltaZ*2

print("Done, new grid is ",coarseSize,"x",coarseSize,"x",coarseSize)

print("Coarsening grid and RHS")
coarseRes = fine2coarse.bringDown(coarseRes,coarseSize,fineSize)
coarseError = fine2coarse.bringDown(coarseError,coarseSize,fineSize)
print("Done")


print("Smoothing current grid") #We need to make sure that the number of iterations is correct since it looks like we may be doing too much relaxation
for i in range(4):
    coarseError = solvers.GaussSolve(coarseSize,coarseError,coarseRes,[coarseDeltaX,coarseDeltaY,coarseDeltaZ])
print("Done")


#----------------------last coarsening and solving-----------------------
print("Calculating 4th coarse grid size")
fineSize = coarseSize
coarseSize = int(fineSize/2 + 1)

fineDeltaX = coarseDeltaX
fineDeltaY = coarseDeltaY
fineDeltaZ = coarseDeltaZ

coarseDeltaX = fineDeltaX*2
coarseDeltaY = fineDeltaY*2
coarseDeltaZ = fineDeltaZ*2

print("Done, new grid is ",coarseSize,"x",coarseSize,"x",coarseSize)

print("Coarsening grid and RHS")
coarseRes = fine2coarse.bringDown(coarseRes,coarseSize,fineSize)
coarseError = fine2coarse.bringDown(coarseError,coarseSize,fineSize)
print("Done")


print("Smoothing current grid")
for i in range(150): #why does python and C have different number of smoothing iterations to get to the same number?
    coarseError = solvers.GaussSolve(coarseSize,coarseError,coarseRes,[coarseDeltaX,coarseDeltaY,coarseDeltaZ])
print("Done")

midpoint  = int((coarseSize-1)/2)
print("midpoint at lowest grid ",coarseError[midpoint,midpoint,midpoint])


#----------------------1st interpolation and solving-----------------------
print("Calculating 1st refined grid size")
fineSize = coarseSize * 2 -1

fineDeltaX = coarseDeltaX/2
fineDeltaY = coarseDeltaY/2
fineDeltaZ = coarseDeltaZ/2

print("Done, new grid is ",fineSize,"x",fineSize,"x",fineSize)

print("Refining and interpolating grid")
fineRes  = coarse2fine.bringUp(coarseRes,fineSize,coarseSize)
fineError = coarse2fine.bringUp(coarseError,fineSize,coarseSize)
print("Done")

midpoint  = int((fineSize-1)/2)



print("Smoothing current grid")
for i in range(2):
    fineError = solvers.GaussSolve(fineSize,fineError,fineRes,[fineDeltaX,fineDeltaY,fineDeltaZ])
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
fineRes  = coarse2fine.bringUp(fineRes,fineSize,coarseSize)
fineError = coarse2fine.bringUp(fineError,fineSize,coarseSize)
print("Done")

for i in range(2):
    fineError = solvers.GaussSolve(fineSize,fineError,fineRes,[fineDeltaX,fineDeltaY,fineDeltaZ])
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
fineRes  = coarse2fine.bringUp(fineRes,fineSize,coarseSize)
fineError = coarse2fine.bringUp(fineError,fineSize,coarseSize)
print("Done")

for i in range(2):
    fineError = solvers.GaussSolve(fineSize,fineError,fineRes,[fineDeltaX,fineDeltaY,fineDeltaZ])
print("Done")

#----------------------3rd interpolation and solving-----------------------
print("Calculating 4th refined grid size")
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
fineRes  = coarse2fine.bringUp(fineRes,fineSize,coarseSize)
fineError = coarse2fine.bringUp(fineError,fineSize,coarseSize)
print("Done")

for i in range(3):
    fineError = solvers.GaussSolve(fineSize,fineError,fineRes,[fineDeltaX,fineDeltaY,fineDeltaZ])
print("Done")

output = grid - fineError #I have no idea why this sum is breaking
trueVal = np.zeros_like(originalGrid)
trueVal = matrixConstructionTools.fillInValues(trueVal,trueFunction,[deltaX,deltaY,deltaZ])

for i in range(3):
    output = solvers.GaussSolve(fineSize,output,rhs,[fineDeltaX,fineDeltaY,fineDeltaZ])
print("Done")

midpoint  = int((fineSize-1)/2)
print("midpoint on modified original grid",output[midpoint,midpoint,midpoint])


trueVal = matrixConstructionTools.buildBoundary(trueVal, boundaries)
diff = trueVal - output
print('REL L2', np.linalg.norm(diff.flatten()) / np.linalg.norm(trueVal.flatten()))
largest = 0 



i=0
while(i < 129):
    j = 0
    while(j < 129):
        k = 0
        while(k < 129):
            if abs(diff[i,j,k]) > largest:
                largestIndex = [i,j,k]
                largest = abs(diff[i,j,k])
            k += 1
        j += 1
    i += 1


print("largest diff",largestIndex,largest)
"""
a = mpl.contourf(np.linspace(0,1,129),np.linspace(0,1,129),output[64])
mpl.show()
b = mpl.contourf(np.linspace(0,1,129),np.linspace(0,1,129),diff[64])
mpl.show()
c = mpl.contourf(np.linspace(0,1,129),np.linspace(0,1,129),trueVal[64])
mpl.show()
d = mpl.contourf(np.linspace(0,1,129),np.linspace(0,1,129),grid[64])
mpl.show()
e = mpl.contourf(np.linspace(0,1,129),np.linspace(0,1,129),rhs[64])
mpl.show()
"""
