////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
	This example demonstrates how to use the Cuda OpenGL bindings to
	dynamically modify a vertex buffer using a Cuda kernel.

	The steps are:
	1. Create an empty vertex buffer object (VBO)
	2. Register the VBO with Cuda
	3. Map the VBO for writing from Cuda
	4. Run Cuda kernel to modify the vertex positions
	5. Unmap the VBO
	6. Render the results using OpenGL

	Host code
*/


#include "Boid.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <shader.h>


// includes, cuda
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include "device_functions.h"; 

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include <vector_functions.h>

#include "DeviceFunctions.h"

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

GLuint translationsVBO;

//vao variables
unsigned int VAO;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

Shader* shPointer;

float2 *pos;

int movementTime = 1;
int timecount = 0;

//boid i is defined by positions[i] and velocities[i]
float2 positions[numberOfBoids];
float2 velocities[numberOfBoids];

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

float2 *dev_positions, *dev_velocities;

const char *windowTitle = "CUDA_Flocking";

__device__ float2 alignment(int threadX, float2 *positions, float2 *velocities, float boidradius)
{
	float2 alignmentVector = make_float2(0, 0);
	int cont = 0;
	for (int i = 0; i < numberOfBoids; i++)
	{
		float2 point1, point2;
		point1 = positions[threadX];
		point2 = positions[i];

		if (threadX != i &&  distanceBetweenPoints(point1, point2) < boidradius)
		{
			alignmentVector.x += velocities[i].x;
			alignmentVector.y += velocities[i].y;
			cont++;
		}
	}

	alignmentVector = vectorDivision(alignmentVector, cont);

	alignmentVector = normalizeVector(alignmentVector); 

	return alignmentVector;
}

__device__ float2 cohesion(int threadX, float2 *positions, float2 *velocities, float boidradius)
{
	float2 cohesionVector = make_float2(0, 0);
	int cont = 0;

	for (int i = 0; i < numberOfBoids; i++)
	{
		float2 point1, point2;
		point1 = positions[threadX];
		point2 = positions[i];
		float distance = sqrtf(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2));

		if (threadX != i &&  distance < boidradius)
		{
			cohesionVector.x += positions[i].x;
			cohesionVector.y += positions[i].y;
			cont++;
		}
	}

	cohesionVector = vectorDivision(cohesionVector, cont); 

	cohesionVector.x -= positions[threadX].x;
	cohesionVector.y -= positions[threadX].y;

	cohesionVector = normalizeVector(cohesionVector); 

	return cohesionVector;
}

__device__ float2 separation(int threadX, float2 *positions, float2 *velocities, float boidradius)
{
	float2 separationVector = make_float2(0, 0);
	int cont = 0;

	for (int i = 0; i < numberOfBoids; i++)
	{
		float2 point1, point2;
		point1 = positions[threadX];
		point2 = positions[i];
		float distance = distanceBetweenPoints(point1, point2); 

		if (threadX != i &&  distance < boidradius)
		{
			separationVector.x += positions[i].x - positions[threadX].x;
			separationVector.y += positions[i].y - positions[threadX].y;
			cont++;
		}

	}

	separationVector.x /= cont;
	separationVector.y /= cont;

	separationVector.x *= -1;
	separationVector.y *= -1;	

	///normalization of separation
	separationVector = normalizeVector(separationVector);

	return separationVector;
}

__global__  void updatePositionsWithVelocities(float2 *positions, float2 *velocities, float boidradius)
{
	unsigned int threadX = blockIdx.x*blockDim.x + threadIdx.x;

	float alignmentWeight, cohesionWeight, separationWeight; 

	alignmentWeight = 9; 
	cohesionWeight = 4;
	separationWeight = 4; 

	float boidSpeed = 0.005;

	if (threadX < numberOfBoids)
	{
		float2 alignmentVector = alignment(threadX, positions, velocities, boidradius) ;
		float2 cohesionVector = cohesion(threadX, positions, velocities, boidradius);
		float2 separationVector = separation(threadX, positions, velocities, boidradius);

		float2 velocityOfTheBoid = velocities[threadX];
		velocityOfTheBoid.x += alignmentVector.x * alignmentWeight 
			+ cohesionVector.x * cohesionWeight 
			+ separationVector.x * separationWeight;

		velocityOfTheBoid.y += alignmentVector.y * alignmentWeight 
			+ cohesionVector.y * cohesionWeight
			+ separationVector.y * separationWeight;

		velocityOfTheBoid = normalizeVector(velocityOfTheBoid);
		velocityOfTheBoid = vectorMultiplication(velocityOfTheBoid, boidSpeed); 
		velocities[threadX] = velocityOfTheBoid;

		positions[threadX].x += velocities[threadX].x;
		positions[threadX].y += velocities[threadX].y;
	}
}

void launchKernel()
{
	updatePositionsWithVelocities << <1, 512 >> > (dev_positions, dev_velocities, boidRadius);
	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

	pArgc = &argc;
	pArgv = argv;

	printf("%s starting...\n", windowTitle);

	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);


	for (int i = 0; i < numberOfBoids; i++)
	{
		int a = rand() % 2 * 2 - 1;
		int b = rand() % 2 * 2 - 1;
		
		velocities[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
		velocities[i] = normalizeVector(velocities[i]); 
		//velocities[i] = make_float2(0.01, 0.01);
		//positions[i] = make_float2((float) 0.0f, (float) 0.0f); 
		positions[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
	}

	createVBO(&vbo);

	cudaMalloc((void**)&dev_positions, numberOfBoids * sizeof(float2));
	cudaMalloc((void**)&dev_velocities, numberOfBoids * sizeof(float2));

	cudaMemcpy(dev_positions, positions, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);

	// start rendering mainloop
	glutMainLoop();

	cudaFree(dev_positions);
	cudaFree(dev_velocities);

	printf("%s completed, returned %s\n", windowTitle, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "CUDA Flock: %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	SDK_CHECK_ERROR_GL();

	Shader shader("shaders/flock.vert", "shaders/flock.frag");
	shader.use();
	shPointer = (Shader*)malloc(sizeof(Shader));
	shPointer = &shader;

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO and fills the VBO so that the positions of the boids can be modifiable
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo)
{
	float quadVertices[] = {
		// positions     // colors
		0.0f,  0.05f,  1.0f, 0.0f, 0.0f,
		0.02f, -0.02f,  0.0f, 1.0f, 0.0f,
		-0.02f, -0.02f,  0.0f, 0.0f, 1.0f
	};

	assert(vbo);

	//vertices vbo
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, vbo);
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_DYNAMIC_DRAW);

	//loading positions of vertices
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	//loading colors
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)3);
	glEnableVertexAttribArray(1);

	//loading position offsets
	glGenBuffers(1, &translationsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, translationsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfBoids, &positions[0], GL_DYNAMIC_DRAW);

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	//this is necessary for instancing in openGL
	glVertexAttribDivisorARB(2, 1);

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback -> called multiple times after the first glutMainLoop() 
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(VAO);

	timecount++;

	if (timecount >= movementTime) {
		launchKernel();
		timecount = 0;
	}

	////loading position offsets
	glBindBuffer(GL_ARRAY_BUFFER, translationsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfBoids, &positions[0], GL_DYNAMIC_DRAW);

	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, numberOfBoids);

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

float2 mouseToWorldCoordinates(int x, int y)
{
	float fX = (float)x / window_width;
	float fY = (float)y / window_width; 
	fX = fX * 2 - 1; 
	fY = -fY * 2 +1; 
	return make_float2(fX, fY); 
}

void setFlockDestination(float2 destination)
{
	for (int i = 0; i < numberOfBoids; i++)
	{
		velocities[i].x = destination.x - positions[i].x; 
		velocities[i].y = destination.y - positions[i].y;
	}
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
}


////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
		float2 destination = mouseToWorldCoordinates(x, y); 
		setFlockDestination(destination); 
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}


void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}