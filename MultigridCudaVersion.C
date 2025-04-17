#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include <omp.h>

#include <windows.h>

#define _USE_MATH_DEFINES

#include <math.h>
int mapCoord(int n,int i, int j, int k){
	return (((n*n)*i) + (n*j) + k);
}

int matrixAdd(int n, double* arrayA, double* arrayB){
	int i,j,k;
	
	for (i=0;i<n;i++){
        for (j=0;j<n;j++){
			for(k=0;k<n;k++){
				arrayA[mapCoord(n,i,j,k)] = arrayA[mapCoord(n,i,j,k)] - arrayB[mapCoord(n,i,j,k)];
			}
		}
	}
	return 1;
	
}

int matrixCopy(n, double* arrayA, double* arrayB){
	int i,j,k;
	
	for (i=0;i<n;i++){
        for (j=0;j<n;j++){
			for(k=0;k<n;k++){
				arrayB[mapCoord(n,i,j,k)] = arrayA[mapCoord(n,i,j,k)]
			}
		}
	}
	return 1;
	
}

__device__ void matrixCopyCUDA(n,double* arrayA, double* arrayB){
	//This can probably be done in parallel
	
	unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	
	arrayB[mapCoord(n,i,j,k)] = arrayA[mapCoord(n,i,j,k)]
	
}

int jacobiIteration(int n, double *u, double *rhs, double deltaX, double deltaY, double deltaZ){
	int i,j,k;
	int nnn = n**3;
	double* copy;
	int m = n-1;
	
	copy = (double*)malloc((nnn) * sizeof(double));
	
	matrixCopy(n,u,copy);
	
	double lambda = deltaX/deltaY;
	double C = 1+pow(lambda,2);	
	
	for (i=1;i<m;i++){
        for (j=1;j<m;j++){
			for(k=1;k<m;k++){
				holder = (pow(lambda,3)*(copy[mapCoord(n,i+1,j,k)]+copy[mapCoord(n,i-1,j,k)])) + (copy[mapCoord(n,i,j+1,k)]+copy[mapCoord(n,i,j-1,k)]) + (copy[mapCoord(n,i,j,k+1)]+copy[mapCoord(n,i,j,k-1)]);
				//printf("sum of X zeroes and Y zeroes: %f\n",holder);
				holder = holder - rhs[mapCoord(n,i,j,k)]*(pow(deltaX,2)); //CHECK this needs to be increased from 2 to 3 since we're in 3D
				//printf("minus rhs: %f\n",holder);
				holder = holder / (3*C); //CHECK if this needs to be increased. I don't believe it does, but CHECK anyway
				//printf("divided by 2 lambda: %f\n ",holder);
				u[mapCoord(n,i,j,k)] = holder;
			}
		}
	}

	
	free(copy);
	return 1
}

//This can be naively done in parallel
__global__ void jacobiIterationCUDA(int n, double *u, double *rhs, double copy*, double deltaX, double deltaY, double deltaZ, int iterations){
	int iterationNum;
	
	int nnn = n**3;
	int m = n-1;


	double lambda = deltaX/deltaY;
	double C = 1+pow(lambda,2);	
	
	unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	
	for(iterationNum=0;iterationNum<iteration;iterationNum++){
	matrixCopyCUDA(n,u,copy);
	
		if (i >= 1 && i < m && j >= 1 && j < m && k >= 1 && k < m){
			holder = (pow(lambda,3)*(copy[mapCoord(n,i+1,j,k)]+copy[mapCoord(n,i-1,j,k)])) + (copy[mapCoord(n,i,j+1,k)]+copy[mapCoord(n,i,j-1,k)]) + (copy[mapCoord(n,i,j,k+1)]+copy[mapCoord(n,i,j,k-1)]);
			holder = holder - rhs[mapCoord(n,i,j,k)]*(pow(deltaX,2)); //CHECK this needs to be increased from 2 to 3 since we're in 3D
			holder = holder / (3*C); //CHECK if this needs to be increased. I don't believe it does, but CHECK anyway
			u[mapCoord(n,i,j,k)] = holder;
		}
	}
	
	//maybe instead we could swap the pointers after the first setup?
	

}

int prettyPrintArray(int n, double* array){
	int i,j,k;
	
	for (i=0;i<n;i++){
        for (j=0;j<n;j++){
			for(k=0;k<n;k++){
				printf("%.2f ", array[mapCoord(n,i,j,k)]);
			}
			printf(";\n");
		}
		printf("\n\n");
	}
	return 1;
}

int prettyPrintSlice(int n, double* array){
	int j,k;
	for (j=0;j<n;j++){
			for(k=0;k<n;k++){
				printf("%.2f ", array[mapCoord(n,1,j,k)]);
			}
			printf(";\n");
		}
}


int dumpArray(int n, double* array){
	int i;
	
	for (i=0;i<pow(n,3);i++){
		printf("%.2f \n", array[i]);
	}
	return 1;
}

//This can be done in parallel
__global__ void zeroOutCUDA(int n, double* matrix){
	int i,j,k;

	unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	
	matrix[mapCoord(n,i,j,k)] = 0.0;


}



int zeroOut(int n, double* matrix){
	int i,j,k;
	for (i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			for(k=0;k<n;k++){
				matrix[mapCoord(n,i,j,k)] = 0.0;
			}
		}
	}
	return 1;
}

										//change this to take a module
int gaussSolveGeneral(int n, double *u, double *rhs, double deltaX, double deltaY, double deltaZ){ //does a SINGLE gauss iteration
	
	int i,j,k;
	int m = n-1;
	
	double holder;

	//double lambda = 2*(deltaX*deltaY);
	
	double lambda = deltaX/deltaY;//this ratio stays the same as we go to smaller grids, even thought deltax and deltay
	double C = 1+pow(lambda,2);		//this ratio is always going to be one because x,y,z all are the same grid of the same length, so we can almost ignore this





	for (i = 1; i < m; i++){ //y index
		for(j = 1; j < m; j++){ //x index   
			for(k = 1;k < m; k++){

			/**
			printf("divDeltX: %f \n",divDeltX);
			printf("divDeltY: %f \n",divDeltY);
			printf("rhs: %f \n",rhs[mapCoord(n,i,j,k)]);
			printf("lambda: %f \n",lambda);
			printf("\n");
			**/
			
							    //double CHECK how this multiplies. Maybe we need to multiply the whole thing by lambda^2
			holder = (pow(lambda,3)*(u[mapCoord(n,i+1,j,k)]+u[mapCoord(n,i-1,j,k)])) + (u[mapCoord(n,i,j+1,k)]+u[mapCoord(n,i,j-1,k)]) + (u[mapCoord(n,i,j,k+1)]+u[mapCoord(n,i,j,k-1)]);
			//printf("sum of X zeroes and Y zeroes: %f\n",holder);
			holder = holder - rhs[mapCoord(n,i,j,k)]*(pow(deltaX,2)); //CHECK this needs to be increased from 2 to 3 since we're in 3D
			//printf("minus rhs: %f\n",holder);
			holder = holder / (3*C); //CHECK if this needs to be increased. I don't believe it does, but CHECK anyway
			//printf("divided by 2 lambda: %f\n ",holder);
			u[mapCoord(n,i,j,k)] = holder;
			
			}
			
		}
		
	}
	
	return 1;
}


int calcResidual(int n, double *u, double *residualArray, double *rhs, double deltaX, double deltaY, double deltaZ){	
	int i,j,k;
	int m = n-1;
				
	
	for (i = 1; i < m; i++){
		for(j = 1; j < m; j++){	//CHECK how to expand this
			for(k=1; k< m; k++){
				residualArray[mapCoord(n,i,j,k)] = (u[mapCoord(n,i+1,j,k)]+u[mapCoord(n,i-1,j,k)]-2*u[mapCoord(n,i,j,k)])/pow(deltaX,2) + ((u[mapCoord(n,i,j+1,k)]+u[mapCoord(n,i,j-1,k)])-2*u[mapCoord(n,i,j,k)])/pow(deltaY,2) + ((u[mapCoord(n,i,j,k+1)]+u[mapCoord(n,i,j,k-1)])-2*u[mapCoord(n,i,j,k)])/pow(deltaX,2) - rhs[mapCoord(n,i,j,k)];
			}
		}
	}
	
	for (i=0; i<n;i++){
		
        
        residualArray[mapCoord(n,m,0,i)] = 0;
        residualArray[mapCoord(n,m,i,0)] = 0;
		residualArray[mapCoord(n,i,0,m)] = 0;
		residualArray[mapCoord(n,i,m,0)] = 0;
		residualArray[mapCoord(n,0,i,m)] = 0;
		residualArray[mapCoord(n,0,m,i)] = 0;
        
    }

		
	return 1;
}



int setUpGuess(int n, double *u, double initGuess){
    int i,j,k;
	int m = n-1;

	for (i=0; i<n;i++){
		
        
        u[mapCoord(n,m,0,i)] = 0;
        u[mapCoord(n,m,i,0)] = 0;
		u[mapCoord(n,i,0,m)] = 0;
		u[mapCoord(n,i,m,0)] = 0;
		u[mapCoord(n,0,i,m)] = 0;
		u[mapCoord(n,0,m,i)] = 0;
        
    }
	
    printf("populating interior with initial guess:");
    for (i=1;i<m;i++){
        for (j=1;j<m;j++){
			for(k = 1;k<m;k++){
				u[mapCoord(n,i,j,k)] = initGuess;
			}
        }
    }
    printf(" done\n");
    
	
	return 1;
}


int getError(int n, double *phi, double *exact){
	int i,j,k;
	
	for (i=0;i<n;i++){
        for (j=0;j<n;j++){
			for (k=0;k<n;k++){
				printf("%f ", phi[mapCoord(n,i,j,k)] - exact[mapCoord(n,i,j,k)]);
			}
			printf(";\n");
		}

		printf("\n");
		printf("\n");
	}
	return 1;
	
}


int setUpRhsDireclet(int n, double *rhs, double inputFunction(double x, double y, double z), double uTop, double uRight, double uBottom, double uLeft, double deltaX, double deltaY, double deltaZ){
	int i,j,k;
	int m = n-1;
	
	//xmin + (deltaX*(j)) = x value here
	//ymin + (deltaY*(i)) = y value here

	int h = 1;
	
	for (i=0; i<n;i++){
		
        
        rhs[mapCoord(n,m,0,i)] = 0;
        rhs[mapCoord(n,m,i,0)] = 0;
		rhs[mapCoord(n,i,0,m)] = 0;
		rhs[mapCoord(n,i,m,0)] = 0;
		rhs[mapCoord(n,0,i,m)] = 0;
		rhs[mapCoord(n,0,m,i)] = 0;
        
    }


	
	
	
	for (i = 1; i < m; i++){
		for(j = 1; j < m; j++){
			for(k = 1; k < m; k++){
				rhs[mapCoord(n,i,j,k)] = inputFunction((deltaY*(i)),(deltaX*(j)),(deltaZ*(k)));
				
				
			}
		}
	}
	
    //printf(" done\n");

	//printf(" done\n");
	return 1;

}

int fineToCoarse(int nFine, int nCoarse, double *fineGrid, double *coarseGrid){
		int i,j,k;
	//we should probably do some kind of weighted tranfer
	
	
	//boundary becomes homogenous
	
	for (i=0;i<nCoarse;i++){
        for (j=0;j<nCoarse;j++){
			for(k=0;k<nCoarse;k++){
				coarseGrid[mapCoord(nCoarse,i,j,k)] = fineGrid[mapCoord(nFine,i*2,j*2,k*2)];
			}
        }
    }
	
	return 1;
}

//this can probably be done in parallel
int fineToCoarseCUDA(){
	
}


int fineToCoarse2(int nFine, int nCoarse, double *fineGrid, double *coarseGrid){
	int i,j,k;
	int mCoarse, mFine;
	
	mCoarse = nCoarse-1;
	mFine = nFine-1;
	//we should probably do some kind of weighted tranfer
	
	
	//boundary becomes homogenous
	
	for (i=1;i< mCoarse;i++){
        for (j=1;j< mCoarse;j++){
			for(k = 1; k < mCoarse;k++){
				
				coarseGrid[mapCoord(nCoarse,i,j,k)] = fineGrid[mapCoord(nFine, 2*i-1,2*j-1,2*k-1)];
				
				// 1,1,1 = 1,1,1
				// 1,1,2 = 1,1,3
				// 1,1,3 = 1,1,5
				// 1,2,3 = 1,3,5
			}
        }
    }
	
	for (i=0; i<nCoarse;i++){
		
        
        coarseGrid[mapCoord(nCoarse,mCoarse,0,i)] = 0;
        coarseGrid[mapCoord(nCoarse,mCoarse,i,0)] = 0;
		coarseGrid[mapCoord(nCoarse,i,0,mCoarse)] = 0;
		coarseGrid[mapCoord(nCoarse,i,mCoarse,0)] = 0;
		coarseGrid[mapCoord(nCoarse,0,i,mCoarse)] = 0;
		coarseGrid[mapCoord(nCoarse,0,mCoarse,i)] = 0;
        
    }
	
	return 1;
}

int coarseToFine(int nFine, int nCoarse, double *fineGrid, double *coarseGrid){
	int i,j,k;

	for (i=0;i<nCoarse;i++){
        for (j=0;j<nCoarse;j++){
			for(k=1;k<nCoarse-1;k++){
				fineGrid[mapCoord(nFine,i*2,j*2,k*2)] = coarseGrid[mapCoord(nCoarse,i,j,k)]; //put in the known points
			}
        }
    }
	
	for (i=2;i<nFine;i+=2){
        for (j=1;j<nFine;j+=2){
			for(k=2;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)])/2; //put in the known points
			}
        }
    }
	
	for (i=1;i<nFine;i+=2){
        for (j=2;j<nFine;j+=2){
			for(k=2;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)])/2; //put in the known points
			}
        }
    }
	
	for (i=2;i<nFine;i+=2){
        for (j=2;j<nFine;j+=2){
			for(k=1;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)])/2; //put in the known points
			}
        }
    }
	
	for (i=2;i<nFine;i+=2){
        for (j=1;j<nFine;j+=2){
			for(k=1;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)] + fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)])/4; //put in the known points
			}
        }
    }
	
	for (i=1;i<nFine;i+=2){
        for (j=2;j<nFine;j+=2){
			for(k=1;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)] + fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)])/4; //put in the known points
			}
        }
    }
	
	for (i=1;i<nFine;i+=2){
        for (j=1;j<nFine;j+=2){
			for(k=2;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)] + fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)])/4; //put in the known points
			}
        }
    }
	
	for (i=1;i<nFine;i+=2){
        for (j=1;j<nFine;j+=2){
			for(k=1;k<nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)] + fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)] + fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)])/6; //put in the known points
			}
        }
    }
	

	/* 

	//three loops for now, probably an easier way of doing it though...
	for (i=1;i<nFine;i+=2){
        for (j=1;j<nFine;j+=2){
           fineGrid[i][j] = (fineGrid[i+1][j+1] + fineGrid[i-1][j+1] + fineGrid[i+1][j-1] + fineGrid[i-1][j-1])/4;
		  
        }
    }
	

	
	for (i=1;i<nFine;i+=2){
        for (j=2;j<nFine-2;j+=2){
           fineGrid[i][j] = (fineGrid[i][j+1] + fineGrid[i][j-1] + fineGrid[i+1][j] + fineGrid[i-1][j])/4;
		  
        }
    }

	
	for (i=2;i<nFine-2;i+=2){
        for (j=1;j<nFine;j+=2){
           fineGrid[i][j] = (fineGrid[i][j+1] + fineGrid[i][j-1] + fineGrid[i+1][j] + fineGrid[i-1][j])/4;
		  
        }
    } */

	return 1;
	
}

//this can maybe be done in parallel? Not sure
int coarseToFineCUDA(){
	
	
}


int coarseToFine2(int nFine, int nCoarse, double *fineGrid, double *coarseGrid){ //it's looking like this guys code is doing the interpolation for us but we should ask around just in case
	int i,j,k;
	int mCoarse, mFine;
	
	mCoarse = nCoarse-1;
	mFine = nFine-1;
	

	for (i=1;i< nCoarse;i++){
        for (j=1;j< nCoarse;j++){
			for(k = 1; k < nCoarse;k++){
				fineGrid[mapCoord(nFine,2*i-1,2*j-1,2*k-1)] = coarseGrid[mapCoord(nCoarse,i,j,k)];
			}
        }
    }




	
	for (i=2;i<nFine+3;i+=2){ // interpolate over i rows in this loop
        for (j=1;j<nFine+3;j+=2){
			for(k = 1; k < nFine+3;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)])/2;
			}
        }
    }
	
	
	for (i=1;i<nFine+3;i+=2){ 
        for (j=2;j<nFine+3;j+=2){// interpolate over j columns in this loop
			for(k = 1; k <nFine+3;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j+1,k)])/2;
			}
        }
    }
	

	
	
	for (i=1;i<nFine+3;i+=2){ 
        for (j=1;j<nFine+3;j+=2){
			for(k = 2; k <=nFine+3;k+=2){ // interpolate over k sheets in this loop
				fineGrid[mapCoord(nFine,i,j,k)] = (fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k+1)])/2;
			}
        }
    }
	

	
	//okay, at this point we should have all the edges, so now we do the faces
	
	for (i=2;i<nFine+3;i+=2){ // interpolate over front faces
        for (j=2;j<nFine+3;j+=2){
			for(k = 1; k < nFine+3;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = ((fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)]) + (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)])) / 4;
			}
        }
    }
	


	
	for (i=1;i<nFine;i+=2){ // interpolate over side faces
        for (j=2;j<nFine;j+=2){
			for(k = 2; k < nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = ((fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)]) + (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)])) / 4;
			}
        }
    }
	

	
	
	for (i=2;i<nFine;i+=2){ // interpolate over plane faces
        for (j=1;j<nFine;j+=2){
			for(k = 2; k < nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = ((fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)]) + (fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)])) / 4;
			}
        }
    }
	
	
	for (i=2;i<nFine;i+=2){ // interpolate over cube centers
        for (j=2;j<nFine;j+=2){
			for(k = 2; k < nFine;k+=2){
				fineGrid[mapCoord(nFine,i,j,k)] = ((fineGrid[mapCoord(nFine,i+1,j,k)] + fineGrid[mapCoord(nFine,i-1,j,k)]) + (fineGrid[mapCoord(nFine,i,j+1,k)] + fineGrid[mapCoord(nFine,i,j-1,k)]) + (fineGrid[mapCoord(nFine,i,j,k+1)] + fineGrid[mapCoord(nFine,i,j,k-1)])) / 6;
			}
        }
    }
	
	
	
	
	

	return 1;
	
	
}

//probably easier to do coarse grid creation in main in the loop



int checkConvergence(int n, double *residualArray, double tolerance){ //this version unfortunatly doesn't require a GOTO. All is lost
	int i,j,k;
	int m;
	
	m = n-1;
	
	for (i=1;i<m;i++){
            for (j=1;j<m;j++){
				for(k = 1; k < m;k++){
					if(residualArray[mapCoord(n,i,j,k)] > tolerance){
						return 0;
					}
				}
            }
        }
	return 1;
	
}

double inputFunction(double x, double y, double z){
	return (-3*pow(M_PI,2))* (sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z));
}

			 //There has to be a 3 because we're doing it over x,y and z

double testAgainstFunction(double x, double y, double z){
	return sin(M_PI*x) * sin(M_PI*y) * sin(M_PI*z);
}



//Dad seems to be of the mind that the relaxation and the update are seperate, but they don't seem to be


int main(){
	
	
	int i,j,k;
	int n, pOf2;
	int coarseGridSizeA,coarseGridSizeB;

	double gridDeltaX, gridDeltaY, gridDeltaZ;
	

	//n = 129;
	n = 129;
						//this only works if we start with odd numbers. If we do even we have to do something else
	coarseGridSizeA = n/2 + 1;	//We really need to CHECK this. For copying the boundaries we can do n/2 - 1 but since we want to only copy the inside and create the boundaries anew we should probably do something more like this
	
	double* rhs;				
	double* u;
	double* residualArray;
	double* errorArray;
	double* checkArray;
	
	
	double* residualCoarseA;
	double* errorCoarseA;

    rhs = (double*)malloc((n*n*n) * sizeof(double)); 
	u = (double*)malloc((n*n*n) * sizeof(double));
	residualArray = (double*)malloc((n*n*n) * sizeof(double));
	errorArray = (double*)malloc((n*n*n) * sizeof(double));
	checkArray = (double*)malloc((n*n*n) * sizeof(double));
	
	
	residualCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	errorCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	

	
	
	double (*func)(double, double, double); 
	double (*otherFunc)(double, double, double); 
	func = inputFunction;
	otherFunc = testAgainstFunction;
	
	double xMax = 1;
	double yMax = 1;
	double xMin = 0;
	double yMin = 0;
	
	double deltaX = (xMax-xMin)/(n-1); 
	double deltaY = (yMax-yMin)/(n-1); 
	double deltaZ = deltaY;
	
	printf("delta x: %f\n", deltaX);
	printf("delta y: %f\n", deltaY);
	
	//-----------------Matrix creation------------
	
	zeroOut(n,rhs);
	zeroOut(n,checkArray);
	
	setUpRhsDireclet(n,rhs,func, 0, 0, 0, 0, deltaX, deltaY, deltaZ);
	setUpRhsDireclet(n,checkArray,otherFunc, 0, 0, 0, 0, deltaX, deltaY, deltaZ);
	zeroOut(n,u);
	zeroOut(n,residualArray);
	zeroOut(coarseGridSizeA,residualCoarseA);
	zeroOut(coarseGridSizeA,errorCoarseA);
	
	
	
	//-----------------First sweep----------------
	
	clock_t begin = clock();
	
	jacobiIterationCUDA(n,u,rhs,deltaX,deltaY,deltaZ,2);
	
	calcResidual(n,u,residualArray,rhs,deltaX,deltaY,deltaZ);

	

	
	printf("top grid output %d  %.4f\n",n, u[mapCoord(n,64,64,64)]);
	
	//-----------------First move and sweep-------
	
	gridDeltaX = deltaX*2; //Check what this needs to be become since the spacing is smaller now
	gridDeltaY = deltaY*2;
	gridDeltaZ = deltaZ*2;
	
	double* level1residual;

	level1residual = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	
	
	zeroOut(coarseGridSizeA,level1residual);
	

	
	fineToCoarse(n, coarseGridSizeA, residualArray, residualCoarseA);
	fineToCoarse(n, coarseGridSizeA, residualArray, level1residual);

	
	
	//no need to move anything else
		
	
	jacobiIterationCUDA(coarseGridSizeA,errorCoarseA,residualCoarseA,copyHolder,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,2);
	
	
	//try without recalculating residualArray
	
	printf("second grid %i, %.4f\n",coarseGridSizeA,errorCoarseA[mapCoord(coarseGridSizeA,32,32,32)]);
	
	//-----------------Next move down------------

	coarseGridSizeB = coarseGridSizeA/2 + 1;
	gridDeltaX = gridDeltaX*2;
	gridDeltaY = gridDeltaY*2;
	gridDeltaZ = gridDeltaZ*2;
	double* residualCoarseB;
	double* errorCoarseB;
	
	double* level2residual;

	level2residual = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	
	
	
	residualCoarseB = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	errorCoarseB = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	

	
	zeroOut(coarseGridSizeB,level2residual);
	
	fineToCoarse(coarseGridSizeA, coarseGridSizeB, residualCoarseA, residualCoarseB);
	fineToCoarse(coarseGridSizeA, coarseGridSizeB, residualCoarseA, level2residual);
	fineToCoarse(coarseGridSizeA, coarseGridSizeB, errorCoarseA, errorCoarseB);
	

	
	
	jacobiIterationCUDA(coarseGridSizeB,errorCoarseB,residualCoarseB,copyHolder,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,2);		

	

	
	
	printf("third grid %i, %.4f\n",coarseGridSizeB,errorCoarseB[mapCoord(coarseGridSizeB,16,16,16)]);
	
	//-----------------Next move down------------
	

	free(residualCoarseA);
	
	
	free(errorCoarseA);
	
	coarseGridSizeA = coarseGridSizeB/2+1;
	gridDeltaX = gridDeltaX*2;
	gridDeltaY = gridDeltaY*2;
	gridDeltaZ = gridDeltaZ*2;
	
	double* level3residual;
	level3residual = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	
	
	residualCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	errorCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	
	
	zeroOut(coarseGridSizeA,level3residual);
	
	fineToCoarse(coarseGridSizeB, coarseGridSizeA, residualCoarseB, residualCoarseA);
	fineToCoarse(coarseGridSizeB, coarseGridSizeA, residualCoarseB, level3residual);
	fineToCoarse(coarseGridSizeB, coarseGridSizeA, errorCoarseB, errorCoarseA);
	

	
	
	
	jacobiIterationCUDA(coarseGridSizeA,errorCoarseA,residualCoarseA,copying,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,4);
	

	
	printf("fourth grid %d %.4f \n", coarseGridSizeA,errorCoarseA[mapCoord(coarseGridSizeA,8,8,8)]);
	
	
	//-----------------Last move down------------
	
	free(residualCoarseB);
	
	free(errorCoarseB);
	
	coarseGridSizeB = coarseGridSizeA/2+1;
	gridDeltaX = gridDeltaX*2;
	gridDeltaY = gridDeltaY*2;
	gridDeltaZ = gridDeltaZ*2;
	                
	residualCoarseB = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	errorCoarseB = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	
	fineToCoarse(coarseGridSizeA, coarseGridSizeB, residualCoarseA, residualCoarseB);
	fineToCoarse(coarseGridSizeA, coarseGridSizeB, errorCoarseA, errorCoarseB);
	
	
	
	
	jacobiIterationCUDA(coarseGridSizeB,errorCoarseB,residualCoarseB,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,80);
	
	
	//calcResidual(coarseGridSizeB,errorCoarseB,residualCoarseB,residualCoarseB,gridDeltaX,gridDeltaY);
	
	//prettyPrintArray(coarseGridSizeB,errorCoarseB);
	printf("final grid %d %.4f \n", coarseGridSizeB, errorCoarseB[mapCoord(coarseGridSizeB,4,4,4)]);
	printf("final residual %f \n", residualCoarseB[mapCoord(coarseGridSizeB,4,4,4)]);
	printf("final delta %f \n", gridDeltaX);


	
	//-----------------Start moving up-----------
	
	free(residualCoarseA);
	
	free(errorCoarseA);
	
	coarseGridSizeA = coarseGridSizeB*2-1;
	gridDeltaX = gridDeltaX/2;
	gridDeltaY = gridDeltaY/2;
	gridDeltaZ = gridDeltaZ/2;
	
	residualCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	errorCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	
	
	zeroOut(coarseGridSizeA,errorCoarseA);
	zeroOut(coarseGridSizeA,residualCoarseA);
	
	coarseToFine(coarseGridSizeA,coarseGridSizeB,errorCoarseA,errorCoarseB);
	coarseToFine(coarseGridSizeA,coarseGridSizeB,residualCoarseA,residualCoarseB);
	
	//prettyPrintArray(coarseGridSizeA,errorCoarseA);
	
	jacobiIterationCUDA(coarseGridSizeA,errorCoarseA,level3residual,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,500);
	
	
	//calcResidual(coarseGridSizeA,errorCoarseA,residualCoarseA,residualCoarseA,gridDeltaX,gridDeltaY);
	
	printf("fourth grid %d %.4f \n", coarseGridSizeA,errorCoarseA[mapCoord(coarseGridSizeA,8,8,8)]);
	printf("fourth grid res %d %.4f \n", coarseGridSizeA,residualCoarseA[mapCoord(coarseGridSizeA,8,8,8)]);
	printf("fourth grid res saved %d %.4f \n", coarseGridSizeA,level3residual[mapCoord(coarseGridSizeA,8,8,8)]);
	printf("fourth grid delta %f \n",gridDeltaX);
	
	

	
	//-----------------Next move up---------------
	
	free(residualCoarseB);
	
	free(errorCoarseB);
	
	coarseGridSizeB = coarseGridSizeA*2-1;
	gridDeltaX = gridDeltaX/2;
	gridDeltaY = gridDeltaY/2;
	gridDeltaZ = gridDeltaZ/2;
	                 
	residualCoarseB = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	errorCoarseB = (double*)malloc((coarseGridSizeB*coarseGridSizeB*coarseGridSizeB) * sizeof(double));
	
	zeroOut(coarseGridSizeB,errorCoarseB);
	zeroOut(coarseGridSizeB,residualCoarseB);
	
	coarseToFine(coarseGridSizeB, coarseGridSizeA, residualCoarseB, residualCoarseA);
	coarseToFine(coarseGridSizeB, coarseGridSizeA, errorCoarseB, errorCoarseA);
	
	for(i=0;i<80;i++){
		gaussSolveGeneral(coarseGridSizeB,errorCoarseB,level2residual,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,);
	}
	//calcResidual(coarseGridSizeB,errorCoarseB,residualCoarseB,residualCoarseB,gridDeltaX,gridDeltaY);
	
		printf("grid size %i\n",coarseGridSizeA);
	int midpoint = coarseGridSizeB-1;
	midpoint = midpoint/2;
	printf("midpoint is %d \n",midpoint);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint,midpoint,midpoint)]);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint+1,midpoint,midpoint)]);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint-1,midpoint,midpoint)]);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint,midpoint+1,midpoint)]);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint,midpoint-1,midpoint)]);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint,midpoint,midpoint+1)]);
	printf("%.4f \n",errorCoarseB[mapCoord(coarseGridSizeB,midpoint,midpoint,midpoint-1)]);
	return 1;

	//-----------------Next move up---------------
	

	free(residualCoarseA);
	
	free(errorCoarseA);
	
	coarseGridSizeA = coarseGridSizeB*2-1;
	gridDeltaX = gridDeltaX/2;
	gridDeltaY = gridDeltaY/2;
	gridDeltaZ = gridDeltaZ/2;
	

	residualCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));
	errorCoarseA = (double*)malloc((coarseGridSizeA*coarseGridSizeA*coarseGridSizeA) * sizeof(double));

	
	zeroOut(coarseGridSizeA,errorCoarseA);
	zeroOut(coarseGridSizeA,residualCoarseA);
	
	coarseToFine(coarseGridSizeA,coarseGridSizeB,errorCoarseA,errorCoarseB);
	coarseToFine(coarseGridSizeA,coarseGridSizeB,residualCoarseA,residualCoarseB);
	
	printf("convereted grid");
	
	
	jacobiIterationCUDA(coarseGridSizeA,errorCoarseA,level1residual,copyHolder,gridDeltaX,gridDeltaY,gridDeltaZ,3);
	
	
	printf("grid size %i\n",coarseGridSizeA);
	midpoint = coarseGridSizeA-1;
	midpoint = midpoint/2;
	printf("midpoint is %d \n",midpoint);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint,midpoint,midpoint)]);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint+1,midpoint,midpoint)]);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint-1,midpoint,midpoint)]);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint,midpoint+1,midpoint)]);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint,midpoint-1,midpoint)]);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint,midpoint,midpoint+1)]);
	printf("%.4f \n",errorCoarseA[mapCoord(coarseGridSizeA,midpoint,midpoint,midpoint-1)]);
	return 1;
	
	printf("second grid %i, %.4f\n",coarseGridSizeA,errorCoarseA[mapCoord(coarseGridSizeA,32,32,32)]);
	
	//-----------------Last move up--------------
	
	zeroOut(n,errorArray);
	coarseToFine(n, coarseGridSizeA, errorArray, errorCoarseA);
	
	printf("first grid %i, %.4f\n",n,errorArray[mapCoord(n,64,64,64)]);

	//-----------------Last move up--------------
	
	printf("%.4f\n",errorArray[mapCoord(n,64,64,64)]);
	printf("%.4f\n",u[mapCoord(n,64,64,64)]);
	printf("expected %.4f\n",checkArray[mapCoord(n,64,64,64)]);
	
	matrixAdd(n,u,errorArray);
	
	printf("result %.4f\n",u[mapCoord(n,64,64,64)]);
	
	
	for(i=0;i<2;i++){
		gaussSolveGeneral(n,u,rhs,deltaX,deltaY,deltaZ);
	}
	
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("took %f",time_spent);
	
}
