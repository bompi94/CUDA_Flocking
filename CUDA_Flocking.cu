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

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

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

GLuint instanceVBO;

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

float2 translations[100];

Shader* shPointer;

int movementTime = 1; 
int timecount = 0; 


#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *windowTitle = "CUDA_Flocking (VBO)";


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

	pArgc = &argc;
	pArgv = argv;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", windowTitle);

	runTest(argc, argv);

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
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard); // il flag indica che CUDA non leggerà dalla risorsa

	// run the cuda part
	runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();


	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}



void massMovement(bool random = false)
{
	int index = 0;
	float offset = 0.001f;

	for (int y = -10; y < 10; y += 2)
	{
		for (int x = -10; x < 10; x += 2)
		{
			float2 translation;
			if (!random) {
				translation.x = (float)x / 10.0f + offset;
				translation.y = (float)y / 10.0f + offset;
			}

			else {
				translation.x = (rand() % 2 * 2 - 1)  * offset;
				translation.y = (rand() % 2 * 2 - 1)  * offset;
			}
			index++; 
			translations[index].x += translation.x;
			translations[index].y += translation.y;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{

	float quadVertices[] = {
		// positions     // colors
		-0.05f,  0.05f,  1.0f, 0.0f, 0.0f,
		0.05f, -0.05f,  0.0f, 1.0f, 0.0f,
		-0.05f, -0.05f,  0.0f, 0.0f, 1.0f
	};
	
	massMovement(true); 

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

	//loading offsets
	glGenBuffers(1, &instanceVBO);
	glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 100, &translations[0], GL_DYNAMIC_DRAW);

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	glVertexAttribDivisorARB(2, 1);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

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

	// run CUDA kernel to generate vertex positions
	//runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(VAO);

	timecount++; 
	if (timecount >= movementTime) {
		massMovement(true); 
		glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 100, &translations[0], GL_DYNAMIC_DRAW);
		timecount = 0;
	}

	glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 100);

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

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
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

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
	if (!d_vbo_buffer)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

		// map buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float *data = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

		// check result
		if (checkCmdLineFlag(argc, (const char **)argv, "regression"))
		{
			// write file for regression test
			sdkWriteFile<float>("./data/regression.dat",
				data, mesh_width * mesh_height * 3, 0.0, false);
		}

		// unmap GL buffer object
		if (!glUnmapBuffer(GL_ARRAY_BUFFER))
		{
			fprintf(stderr, "Unmap buffer failed.\n");
			fflush(stderr);
		}

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard));

		SDK_CHECK_ERROR_GL();
	}
}
