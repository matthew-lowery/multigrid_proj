import numpy as np

#this function moves through the array and sets the boundary to the relevant values
#this assumes diriclet
def buildBoundary(grid,boundary):
    edge = grid.shape[0]
    print("--Filling in grid boundary points")
    print("==Input size:",edge," Boundary values:",boundary)
    i = 0
    while(i<edge):
        j = 0
        while(j<edge):
            grid[i,j,0] = boundary[0]
            grid[i,j,edge-1]  = boundary[1]
            grid[0,i,j] = boundary[2]
            grid[edge-1,i,j] = boundary[3]
            grid[i,0,j] = boundary[4]
            grid[i,edge-1,j] = boundary[5]

            j+=1
        i+=1
    print("--Done")
    return grid

#this is what we run when we move from grid to grid. Since we're assuming boundary, we can drop the residual to zero I think
def HandleMovedBoundary(grid):
    print("----Adjusting boundary for residual")

    edge = grid.shape[0] -1
    print("====Boundary location:",edge)

    grid[0,edge,:] = 0
    grid[edge,0,:] = 0
    grid[0,:,edge] = 0
    grid[edge,:,0] = 0
    grid[:,edge,0] = 0
    grid[:,0,edge] = 0
        
    print("----Done")
    return 0

def fillInValues(grid,function,deltaXYZ):
    innerEdge = grid.shape[0]
    print("--Populating",innerEdge**3,"internal grid points")
    print("==Input size:",innerEdge," DeltaX/Y/Z:",deltaXYZ)
    i = 1
    while(i<innerEdge):
        j = 1
        while(j<innerEdge):
            k = 1
            while(k<innerEdge):
                grid[i,j,k] = function(i*deltaXYZ[0],j*deltaXYZ[1],k*deltaXYZ[2])                 
                k+=1
            j+=1
        i+=1
    print("--Done")
    return grid

#we may actually just be able to do this with a mesh
    