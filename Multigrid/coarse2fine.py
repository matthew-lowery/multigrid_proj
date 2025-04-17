import numpy as np

def bringUp(coarseGrid,newSize,oldSize):

    fineGrid = np.zeros((int(newSize),int(newSize),int(newSize)))

    print(newSize,oldSize,coarseGrid.shape,fineGrid.shape)

    
    i = 0
    while(i < oldSize):
        j = 0
        while(j < oldSize):
            k = 1
            while(k < oldSize):
                fineGrid[i*2,j*2,k*2] = coarseGrid[i,j,k]
                k+=1
            j+=1
        i+=1

    i = 2
    while(i < newSize):
        j = 1
        while(j < newSize):
            k = 2
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i,j+1,k] + fineGrid[i,j-1,k])/2
                k+=2
            j+=2
        i+=2

    i = 1
    while(i < newSize):
        j = 2
        while(j < newSize):
            k = 2
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i+1,j,k] + fineGrid[i-1,j,k])/2
                k+=2
            j+=2
        i+=2

    i = 2
    while(i < newSize):
        j = 2
        while(j < newSize):
            k = 1
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i,j,k+1] + fineGrid[i,j,k-1])/2
                k+=2
            j+=2
        i+=2

    i = 2
    while(i < newSize):
        j = 1
        while(j < newSize):
            k = 1
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i,j,k+1] + fineGrid[i,j,k-1] + fineGrid[i,j+1,k] + fineGrid[i,j-1,k])/4
                k+=2
            j+=2
        i+=2

    i = 1
    while(i < newSize):
        j = 2
        while(j < newSize):
            k = 1
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i,j,k+1] + fineGrid[i,j,k-1] + fineGrid[i+1,j,k] + fineGrid[i-1,j-1,k])/4
                k+=2
            j+=2
        i+=2


    i = 1
    while(i < newSize):
        j = 1
        while(j < newSize):
            k = 2
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i,j+1,k] + fineGrid[i,j-1,k] + fineGrid[i+1,j,k] + fineGrid[i-1,j,k])/4
                k+=2
            j+=2
        i+=2

    i = 1
    while(i < newSize):
        j = 1
        while(j < newSize):
            k = 1
            while(k < newSize):
                fineGrid[i,j,k] = (fineGrid[i,j+1,k] + fineGrid[i,j-1,k] + fineGrid[i+1,j,k] + fineGrid[i-1,j,k] + fineGrid[i,j,k+1] + fineGrid[i,j,k-1])/6
                k+=2
            j+=2
        i+=2



    return fineGrid