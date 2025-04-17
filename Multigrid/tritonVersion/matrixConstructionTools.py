import numpy as np

#this function moves through the array and sets the boundary to the relevant values
#this assumes diriclet
def buildBoundary(grid,boundary):
    edge = np.size(grid)
    i = 0
    j = 0

    while(i<edge):
        while(j<edge):
            grid[i,j,0] = boundary[0]
            grid[i,j,edge-1]  = boundary[1]
            grid[0,i,j] = boundary[2]
            grid[edge-1,i,j] = boundary[3]
            grid[i,0,j] = boundary[4]
            grid[i,edge-1,j] = boundary[5]

            j+=1
        i+=1

    return grid

#this is what we run when we move from grid to grid. Since we're assuming boundary, we can drop the residual to zero I think
def HandleMovedBoundary(grid):
    edge = np.size(grid) -1

    grid[0,edge,:] = 0
    grid[edge,0,:] = 0
    grid[0,:,edge] = 0
    grid[edge,:,0] = 0
    grid[:,edge,0] = 0
    grid[:,0,edge] = 0
        
    return 0

def fillInValues(grid,function,deltaXYZ):
    innerEdge = np.size(grid)
    i,j,k = 1 #change this to use the scale
    while(i<innerEdge):
        while(j<innerEdge):
            while(k<innerEdge):
                grid[i,j,k] = function(i*deltaXYZ,j*deltaXYZ,k*deltaXYZ)                 
                k+=1
            j+=1
        i+=1

    return grid

#we may actually just be able to do this with a mesh
    