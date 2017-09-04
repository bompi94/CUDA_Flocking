#pragma once
#ifndef CUDAFLOCKING_H
#define CUDAFLOCKING_H

// Utilities and timing functions
#include <vector_types.h>
#include <vector_functions.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <shader.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <cuda_gl_interop.h>
#include "device_functions.h"; 
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


#include "Boid.h"
#include "Cell.h"



class CUDAFlocking
{
public:
	static const int numStreams = 9;
	static const int boidPerThread = 5;

	CUDAFlocking(); 
	void init(); 
	void calculateBoidsPositions();
	void setFlockDestination(float2 destination);
	float2* getPositions();
	float2* getObstacleCenters(); 
	float* getObstacleRadii(); 

private:
	//boid i is defined by positions[i] and velocities[i]
	float2* positions;
	float2* velocities;
	float2 *dev_positions, *dev_velocities;

	Cell* cells;
	Cell* dev_cells;
	//array that associates cells with the last boid to be registered in it
	//if i take cell 
	int* cellHead;
	int* dev_cellHead;
	//array that represents a chain of boids, every index is corresponding to the boid index and in every 
	//cell there is the index of the next boid
	int* cellNext;
	int* dev_cellNext;
	//array that associates a cell with its neighbours indices
	//every cell has 8 neighbours
	//to find the neighbours of cell x you have to index 
	//neighbour[x+i], i goes from 0 to 7
	int** neighbours;
	int* dev_neighbours;
	//remembers the cell every boid is in 
	int* dev_boidXCellsIDs;
	//stores the flocking vectors in between kernels
	float2* dev_temp;
	cudaStream_t streams[9];
	int offset = numberOfBoids / numStreams;

	float2* obstacleCenters;
	float* obstacleRadii;

	void preparePositionsAndVelocitiesArray();
	void prepareCells();
	void prepareObstacles();
	void prepareBoidCUDADataStructures();
	void prepareObstaclesCUDADataStructures();
	void prepareCellsCUDADataStructures();
	void freeCUDADataStructures();
};
#endif //CUDAFLOCKING_H