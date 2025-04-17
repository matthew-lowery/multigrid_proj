import numpy as np
import matrixConstructionTools

def calcResidual(size,residualArray,u,rhs,deltas):

    m = size-1

    i = 1
    while(i<m):
        j = 1
        while(j<m): #careful, I can't remember how while handles things so be sure to get all values
            k = 1
            while(k<m):
                residualArray[i,j,k] = (u[i+1,j,k]+u[i-1,j,k]-2*u[i,j,k])/pow(deltas[0],2) + ((u[i,j+1,k]+u[i,j-1,k])-2*u[i,j,k])/pow(deltas[1],2) + ((u[i,j,k+1]+u[i,j,k-1])-2*u[i,j,k])/pow(deltas[2],2) - rhs[i,j,k]
                k+=1
            j+=1
        i+=1


    matrixConstructionTools.HandleMovedBoundary(residualArray)
    print("--Done")

    return residualArray


def calcError():
    #check value against correct value



    return 0



def checkConvergence():

    #chef's choice



    return 0