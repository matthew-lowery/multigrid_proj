import numpy as np

def bringUp(grid,newSize,oldSize):
    fineGrid = np.zeros((newSize,newSize,newSize))
    i,j,k = 0

    while(i<oldSize):
        while (j<oldSize):
            while (k<oldSize):
                fineGrid[i*2,j*2,k*2] = grid[i,j,k]
                k+=1
            j+=1
        i+=1
    
    i = 2
    j = 1
    k = 2
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i,j+1,k] + fineGrid[i,j-1,k])/2
                k+=2
            j+=2
        i+=2			

    i = 1
    j = 2
    k = 2
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i+1,j,k] + fineGrid[i-1,j,k])/2
                k+=2
            j+=2
        i+=2		
        
    i = 2
    j = 2 
    k = 1
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i,j,k+1] + fineGrid[i,j,k-1])/2
                k+=2
            j+=2
        i+=2

    i = 2
    j = 1
    k = 1
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i,j,k+1] + fineGrid[i,j,k-1] + fineGrid[i,j+1,k] + fineGrid[i,j-1,k])/4
                k+=2
            j+=2
        i+=2	

    i = 1
    j = 2
    k = 1
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i,j,k+1] + fineGrid[i,j,k-1] + fineGrid[i+1,j,k] + fineGrid[i-1,j,k])/4
                k+=2
            j+=2
        i+=2	
    
    i = 1
    j = 1
    k = 2
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i,j+1,k] + fineGrid[i,j-1,k] + fineGrid[i+1,j,k] + fineGrid[i-1,j,k])/4
                k+=2
            j+=2
        i+=2

    i = 1
    j = 1
    k = 1
    while(i<newSize):
        while (j<newSize):
            while (k<newSize):
                fineGrid[i,j,k] = (fineGrid[i,j+1,k] + fineGrid[i,j-1,k] + fineGrid[i+1,j,k] + fineGrid[i-1,j,k] + fineGrid[i,j,k+1] + fineGrid[i,j,k-1])/6
                k+=2
            j+=2
        i+=2



    return fineGrid