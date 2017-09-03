#pragma once
#ifndef CUDAFLOCKING_H
#define CUDAFLOCKING_H

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_gl.h>
#include <GL/freeglut.h>

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

#include "Boid.h"
#include "DeviceFunctions.h"

#ifndef _CELL_H
#include "Cell.h"
#endif

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#define MAX(a,b) ((a > b) ? a : b)

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

const unsigned int window_width = 512;
const unsigned int window_height = 512;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
StopWatchInterface *timer = NULL;
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
GLuint translationsVBO;
unsigned int VAO;
int *pArgc = NULL;
char **pArgv = NULL;
float2 *pos;
int movementTime = 1;
int timecount = 0;

const int numStreams = 9;
const int boidPerThread = 5;

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void timerEvent(int value);
void cleanup();
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
void computeFPS();
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

	if (boidIndex < numberOfBoids) 
	{
		int cellID = boidXCellsIds[boidIndex];
		int base = boidIndex * 4;

		float2 boidVelocity = calculateBoidVelocity(velocities[boidIndex],
			temp[base + 0], temp[base + 1], temp[base + 2], temp[base + 3]);
		boidVelocity = normalizeVector(boidVelocity);
		boidVelocity = vectorMultiplication(boidVelocity, 0.003);

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
	if (cells[myCellID].IsPositionInCell(pos))
		return cells[myCellID].id;


	//changing cell, search the next in the neighbours of my cell
	for (int i = 0; i < 8; i++)
	{
		int base = myCellID * 8;
		int currentNeighbourIndex = neighbours[base + i];
		Cell currentNeighbour = cells[currentNeighbourIndex];
		if (currentNeighbour.IsPositionInCell(pos))
			return currentNeighbour.id;
	}
	printf("oh no\n");
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