import numpy as np


def bringDown(grid,newSize,oldSize):
    coarseGrid = np.zeros((newSize,newSize,newSize))
    i,j,k = 0

    while(i<newSize):
        while(j<newSize):
            while(k<newSize):
                coarseGrid[i,j,k] = grid[i*2,j*2,k*2]

                k+=1
            j+=1
        i+=1

    return coarseGrid