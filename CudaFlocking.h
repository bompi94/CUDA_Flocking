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
void prepareCUDADataStructures();
void freeCUDADataStructures();
void endApplication(); 
void computeFPS(); 
__global__  void updatePositionsWithVelocities1(float2 *positions, float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii);
float2 mouseToWorldCoordinates(int x, int y);
void setFlockDestination(float2 destination);
void sendFlockToMouseClick(int x, int y);
void prepareObstacles(); 
__device__ void screenOverflow(float2 *positions, int boidIndex);
void prepareBoidCUDADataStructures(); 
void prepareObstaclesCUDADataStructures();

__global__  void updatePositionsWithVelocities1(float2 *positions, 
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if (boidIndex < numberOfBoids)
	{
		float2 alignmentVector = alignment(boidIndex, positions, velocities, boidradius);
		float2 cohesionVector = cohesion(boidIndex, positions, velocities, boidradius);
		float2 separationVector = separation(boidIndex, positions, velocities, boidradius);
		float2 obstacleAvoidanceVector = obstacleAvoidance(positions[boidIndex], velocities[boidIndex], obstacleCenters, obstacleRadii);
		velocities[boidIndex] = calculateBoidVelocity(velocities[boidIndex], alignmentVector,
			cohesionVector, separationVector, obstacleAvoidanceVector);
		positions[boidIndex].x += velocities[boidIndex].x;
		positions[boidIndex].y += velocities[boidIndex].y;
		screenOverflow(positions, boidIndex);
	}
}

__global__ void updatePositionsWithVelocities2(float2 *positions, 
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii)
{
	unsigned int threadX = blockDim.x*blockIdx.x + threadIdx.x; 
	unsigned int threadY = threadIdx.y; 
	float boidSpeed = 0.003;
	float2 vector; 
	float weight;

	if (threadX < numberOfBoids) {
		if (threadY == 0) { //allineamento 
			vector = alignment(threadX, positions, velocities, boidradius);
			weight = 100;
		}
		if (threadY == 1) {//coesione
			vector = cohesion(threadX, positions, velocities, boidradius);
			weight = 100;
		}
		if (threadY == 2) {//separazione
			vector = separation(threadX, positions, velocities, boidradius);
			weight = 101;
		}
		if (threadY == 3) {//ostacoli
			vector = obstacleAvoidance(positions[threadX], velocities[threadX], obstacleCenters, obstacleRadii);
			weight = 100;
		}

		velocities[threadX].x += vector.x * weight;
		velocities[threadX].y += vector.y * weight;

		if (threadY == 3) {
			velocities[threadX] = normalizeVector(velocities[threadX]);
			velocities[threadX] = vectorMultiplication(velocities[threadX], boidSpeed);
			positions[threadX].x += velocities[threadX].x;
			positions[threadX].y += velocities[threadX].y;
			screenOverflow(positions, threadX);
		}
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

#endif //CUDAFLOCKING_H