import numpy
import scipy 

#for now it's in 3d, but we can shift it up or down
def GaussSolve(n,u,rhs,deltas):

    m = n-1
    holder = 0

    lambdaVal = deltas[0]/deltas[1]
    C = 1 + lambdaVal**2

    #print("--Performing Gauss solve")
    i = 0
    while(i<m):
        j = 0
        while(j<m):
            k = 0
            while(k<m):
                holder = (lambdaVal**3) * (u[i+1,j,k]+u[i-1,j,k]) + (u[i,j+1,k]+u[i,j-1,k]) + (u[i,j,k+1]+u[i,j,k-1])
                holder = holder - (rhs[i,j,k]*(deltas[1]**2))
                holder = holder / (3*C)
                u[i,j,k] = holder
                k += 1
            j += 1
        i += 1

    #print("--Done")
    return u


#this one can be made parallel easily
def JacobiSolve(n,u,rhs,deltas):
    i = 0
    m = n-1 
    holder = 0
    uCopy = u #this way we don't overwrite values we need to use later

    lambdaVal = deltas[0]/deltas[1]
    C = 1 + lambdaVal**2

    while(i<m):
        j = 0
        while(j<m):
            k = 0
            while(k<m):
                holder = (lambdaVal**3) * (u[i+1,j,k]+u[i-1,j,k]) + (u[i,j+1,k]+u[i,j-1,k]) + (u[i,j,k+1]+u[i,j,k-1])
                holder = holder - (rhs[i,j,k]*(deltas[0]**2))
                holder = holder / (3*C)
                uCopy[i,j,k] = holder
                k += 1
            j += 1
        i += 1

    return uCopy



