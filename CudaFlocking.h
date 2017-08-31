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
//boid i is defined by positions[i] and velocities[i]
float2 positions[numberOfBoids];
float2 velocities[numberOfBoids];
float2 *dev_positions, *dev_velocities;

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
void prepareCUDADataStructures();
void freeCUDADataStructures();
void endApplication();
void computeFPS();
__global__  void updatePositionsWithVelocities1(float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int numberOfCells,
	int* cellHead, int* cellNext, int* neighbours);
float2 mouseToWorldCoordinates(int x, int y);
void setFlockDestination(float2 destination);
void sendFlockToMouseClick(int x, int y);
void prepareObstacles();
__device__ void screenOverflow(float2 *positions, int boidIndex);
void prepareBoidCUDADataStructures();
void prepareObstaclesCUDADataStructures();
void prepareCellsCUDADataStructures();
__device__ void cellsSetup(float2 *positions,
	Cell* cells, int numberOfCells,
	int* cellHead, int* cellNext);
__device__ void cellsResetAndUpdatePosition(float2 *positions, float2* velocities,
	Cell* cells, int numberOfCells,
	int* cellHead, int* cellNext);

__device__ int GetCellId(Cell* cells, float2 pos, int numberOfCells);

#if __CUDA_ARCH__ >= 200 //necessary to compile atomicExch
__global__  void updatePositionsWithVelocities1(float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int numberOfCells,
	int* cellHead, int* cellNext, int* neighbours)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int boidY = threadIdx.y;

	if (boidIndex < numberOfBoids)
	{
		int cellID = GetCellId(cells, positions[boidIndex], numberOfCells);

		if (cellID != -1) 
		{

			if (boidY == 8) {
				int lastStartElement = atomicExch(&cellHead[cellID], boidIndex);
				cellNext[boidIndex] = lastStartElement;
			}

			int neighbourCellID = neighbours[(cellID * 8) + boidY % 8];
			int neighbourBoidIndex = cellHead[neighbourCellID];

			if (boidY == 8)
			{
				neighbourCellID = cellID;
				neighbourBoidIndex = boidIndex;
			}

			if (neighbourBoidIndex != -1) {

				float2 alignmentVector = alignment(neighbourBoidIndex, positions, velocities, boidradius, cellNext);
				float2 cohesionVector = cohesion(boidIndex, neighbourBoidIndex, positions, velocities, boidradius, cellNext);
				float2 separationVector = separation(boidIndex, neighbourBoidIndex, positions, velocities, boidradius, cellNext);

				float2 obstacleAvoidanceVector = obstacleAvoidance(positions[boidIndex], velocities[boidIndex], obstacleCenters, obstacleRadii);

				float2 boidVelocity = calculateBoidVelocity(velocities[boidIndex], alignmentVector,
					cohesionVector, separationVector, obstacleAvoidanceVector);

				float2 v = velocities[boidIndex];
				v = vectorSum(v, boidVelocity);
				v = normalizeVector(v);
				v = vectorMultiplication(v, 0.003); //multiply by boidSpeed

				velocities[boidIndex] = v;
				//atomicExch(&velocities[boidIndex].x, v.x);
				//atomicExch(&velocities[boidIndex].y, v.y);
			}

			if (boidY == 8) 
			{

				float2 vv = positions[boidIndex];
				vv = vectorSum(vv, velocities[boidIndex]);
				positions[boidIndex] = vv; 

				//atomicExch(&positions[boidIndex].x, vv.x);
				//atomicExch(&positions[boidIndex].y, vv.y);

				screenOverflow(positions, boidIndex);

				cellNext[boidIndex] = -1;
				cellHead[cellID] = -1;
			}
		}
	}
}
#endif

#if __CUDA_ARCH__ >= 200 //necessary to compile atomicExch
__device__ void cellsSetup(float2 *positions,
	Cell* cells, int numberOfCells,
	int* cellHead, int* cellNext)
{

	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if (boidIndex < numberOfBoids)
	{
		int cellID = GetCellId(cells, positions[boidIndex], numberOfCells);
		int lastStartElement = atomicExch(&cellHead[cellID], boidIndex);
		cellNext[boidIndex] = lastStartElement;
	}
}
#endif


__device__ void cellsResetAndUpdatePosition(float2 *positions, float2* velocities,
	Cell* cells, int numberOfCells,
	int* cellHead, int* cellNext)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x;

	if (boidIndex < numberOfBoids) {

		positions[boidIndex].x += velocities[boidIndex].x;
		positions[boidIndex].y += velocities[boidIndex].y;
		screenOverflow(positions, boidIndex);

		int cellID = GetCellId(cells, positions[boidIndex], numberOfCells);
		cellNext[boidIndex] = -1;
		cellHead[cellID] = -1;
	}
}


__device__ void screenOverflow(float2 *positions, int boidIndex)
{
	float limit = 0.99;
	if (positions[boidIndex].x > limit || positions[boidIndex].x < -limit)
	{
		positions[boidIndex].x *= -1;
	}
	if (positions[boidIndex].y > limit || positions[boidIndex].y < -limit)
	{
		positions[boidIndex].y *= -1;
	}
}

__device__ int GetCellId(Cell* cells, float2 pos, int numberOfCells)
{
	for (int i = 0; i < numberOfCells*numberOfCells; i++)
	{
		if (cells[i].IsPositionInCell(pos))
			return cells[i].id;
	}

	return -1;
}

#endif //CUDAFLOCKING_H