import numpy as np


def bringDown(grid,newSize,oldSize):
    coarseGrid = np.zeros((int(newSize),int(newSize),int(newSize)))
    
    i = 0
    while(i < newSize):
        j = 0
        while(j < newSize):
            k = 0
            while(k < newSize):
                coarseGrid[i,j,k] = grid[i*2,j*2,k*2]
                k+=1
            j+=1
        i+=1

    return coarseGrid