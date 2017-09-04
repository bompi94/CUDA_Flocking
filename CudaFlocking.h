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
#include "DeviceFunctions.h"
#include "Graphics.h"
#include "Helper.h"
#include "Cell.h"


int g_Index = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
GLuint vbo;
void *d_vbo_buffer = NULL;
GLuint translationsVBO;
int *pArgc = NULL;
char **pArgv = NULL;
float2 *pos;
int movementTime = 1;
int timecount = 0;

//----important variables
const int numStreams = 9;
const int boidPerThread = 5;
Graphics graphics;
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

//boid i is defined by positions[i] and velocities[i]
float2* positions;
float2* velocities;
float2 *dev_positions, *dev_velocities;

cudaStream_t streams[numStreams];
int offset = numberOfBoids / numStreams;

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void prepareGraphicsToRenderBoids(GLuint *vbo);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
void startApplication(int argc, char ** argv);
void endApplication();
void calculateBoidsPositions();
void registerGlutCallbacks();
void preparePositionsAndVelocitiesArray();
void prepareCells();
void freeCUDADataStructures();
void endApplication();
__global__  void computeFlocking(float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int* cellHead, int* cellNext, int* neighbours, float2* temp, int* boidXCellsIDs, int offset);
float2 mouseToWorldCoordinates(int x, int y);
void setFlockDestination(float2 destination);
void sendFlockToMouseClick(int x, int y);
void prepareObstacles();
__device__ void screenOverflow(float2 *positions, int boidIndex);
void prepareBoidCUDADataStructures();
void prepareObstaclesCUDADataStructures();
void prepareCellsCUDADataStructures();
__device__ int GetCellId(int cellID, int* neighbours, Cell* cells, float2 pos, int numberOfCells);


__global__ void setupCells(float2 *positions, int* cellHead, int* cellNext, Cell* cells, int numberOfCells, int* boidXCellsIDs, int* neighbours, int streamNumber)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x + streamNumber*numberOfBoids / numStreams;
	if (boidIndex < numberOfBoids) {

		int cellID = GetCellId(boidXCellsIDs[boidIndex], neighbours, cells, positions[boidIndex], numberOfCells);

		int lastStartElement = cellHead[cellID];
		cellHead[cellID] = boidIndex;
		cellNext[boidIndex] = lastStartElement;
		boidXCellsIDs[boidIndex] = cellID;
	}
}

__global__ void makeMovement(float2* positions, float2* velocities,
	int*  cellHead, int* cellNext, Cell* cells, int numberOfCells, float2* temp, int* boidXCellsIds, int  streamNumber)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x + streamNumber*numberOfBoids / numStreams;
	const float boidSpeed = 0.002f;
	if (boidIndex < numberOfBoids)
	{
		int cellID = boidXCellsIds[boidIndex];
		int base = boidIndex * 4;

		float2 boidVelocity = calculateBoidVelocity(velocities[boidIndex],
			temp[base + 0], temp[base + 1], temp[base + 2], temp[base + 3]);
		boidVelocity = normalizeVector(boidVelocity);
		boidVelocity = vectorMultiplication(boidVelocity, boidSpeed);

		velocities[boidIndex].x = boidVelocity.x;
		velocities[boidIndex].y = boidVelocity.y;

		positions[boidIndex] = vectorSum(positions[boidIndex], velocities[boidIndex]);
		screenOverflow(positions, boidIndex);

		cellNext[boidIndex] = -1;
		cellHead[cellID] = -1;
	}
}

__device__ void computeAllForBoid(int boidIndex, int boidY, float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int numberOfCells, int* cellHead, int* cellNext, int* neighbours, float2* temp, int* boidXCellsIDs)
{

	if (boidXCellsIDs[boidIndex] != -1)
	{
		int neighbourCellID = neighbours[(boidXCellsIDs[boidIndex] * 8) + boidY % 8];

		if (boidY == 8)
		{
			neighbourCellID = boidXCellsIDs[boidIndex];;
		}

		if (cellHead[neighbourCellID] != -1)
		{
			temp[boidIndex * 4 + 0] = normalizeVector(vectorSum(temp[boidIndex * 4 + 0], alignment(cellHead[neighbourCellID], positions, velocities, boidradius, cellNext)));
			temp[boidIndex * 4 + 1] = normalizeVector(vectorSum(temp[boidIndex * 4 + 1], cohesion(boidIndex, cellHead[neighbourCellID], positions, velocities, boidradius, cellNext)));
			temp[boidIndex * 4 + 2] = normalizeVector(vectorSum(temp[boidIndex * 4 + 2], separation(boidIndex, cellHead[neighbourCellID], positions, velocities, boidradius, cellNext)));
			temp[boidIndex * 4 + 3] = normalizeVector(obstacleAvoidance(positions[boidIndex], velocities[boidIndex], obstacleCenters, obstacleRadii));
		}

	}
}

__global__  void computeFlocking(float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int* cellHead, int* cellNext, int* neighbours, float2* temp, int* boidXCellsIDs, int streamNumber)
{
	unsigned int boidIndex = (blockIdx.x*blockDim.x + threadIdx.x) * boidPerThread;

	for (size_t i = 0; i < boidPerThread; i++)
	{
		int index = boidIndex + i;
		if (index < numberOfBoids)
		{
			computeAllForBoid(index, streamNumber, positions, velocities, boidradius, obstacleCenters, obstacleRadii,
				cells, numberOfCells, cellHead, cellNext, neighbours, temp, boidXCellsIDs);
		}
	}
}


__device__ int GetCellId(int myCellID, int* neighbours, Cell* cells, float2 pos, int numberOfCells)
{
	//don't have a cell yet, so i search it through all the cells
	if (myCellID == -1)
	{
		for (int i = 0; i < numberOfCells*numberOfCells; i++)
		{
			if (cells[i].IsPositionInCell(pos))
				return cells[i].id;
		}
	}

	//check if I'm still in my cell
	else if (cells[myCellID].IsPositionInCell(pos))
		return cells[myCellID].id;

	else
		//changing cell, search the next in the neighbours of my cell
		for (int i = 0; i < 8; i++)
		{
			int base = myCellID * 8;
			int currentNeighbourIndex = neighbours[base + i];
			Cell currentNeighbour = cells[currentNeighbourIndex];
			if (currentNeighbour.IsPositionInCell(pos))
				return currentNeighbour.id;
		}
	return -1;
}

__device__ void screenOverflow(float2 *positions, int boidIndex)
{
	float limit = 1;
	if (positions[boidIndex].x >= limit || positions[boidIndex].x <= -limit)
	{
		if (positions[boidIndex].x > 0)
			positions[boidIndex].x = limit - 0.001;
		if (positions[boidIndex].x < 0)
			positions[boidIndex].x = -limit + 0.001;
		positions[boidIndex].x *= -1;
	}
	if (positions[boidIndex].y >= limit || positions[boidIndex].y <= -limit)
	{
		if (positions[boidIndex].y > 0)
			positions[boidIndex].y = limit - 0.001;
		if (positions[boidIndex].y < 0)
			positions[boidIndex].y = -limit + 0.001;
		positions[boidIndex].y *= -1;
	}
}

#endif //CUDAFLOCKING_H