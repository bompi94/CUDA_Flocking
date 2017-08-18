#include "Utilities.h"

int main(int argc, char **argv)
{
	startApplication(argc, argv);
	glutMainLoop();
	endApplication();
}

void startApplication(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;
	printf("%s starting...\n", windowTitle);
	sdkCreateTimer(&timer);
	initGL(&argc, argv);
	registerGlutCallbacks();
	preparePositionsAndVelocitiesArray();
	createVBO(&vbo);
	prepareCUDADataStructures();
}

bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
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

void registerGlutCallbacks()
{
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutCloseFunc(cleanup);
}

void preparePositionsAndVelocitiesArray()
{
	for (int i = 0; i < numberOfBoids; i++)
	{
		int a = randomMinusOneOrOneInt();
		int b = randomMinusOneOrOneInt();
		velocities[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
		velocities[i] = normalizeVector(velocities[i]);
		positions[i] = make_float2(randomMinusOneOrOneFloat(), randomMinusOneOrOneFloat());
	}
}

void createVBO(GLuint *vbo)
{
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

void prepareCUDADataStructures()
{
	cudaMalloc((void**)&dev_positions, numberOfBoids * sizeof(float2));
	cudaMemcpy(dev_positions, positions, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_velocities, numberOfBoids * sizeof(float2));
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
}

void display()
{
	sdkStartTimer(&timer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(VAO);
	timecount++;
	if (timecount >= movementTime) {
		launchFlockingKernel();
		timecount = 0;
	}
	loadPositionOffsetOnVBO();
	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, numberOfBoids);
	glutSwapBuffers();
	sdkStopTimer(&timer);
	computeFPS();
}

void launchFlockingKernel()
{
	updatePositionsWithVelocities << <1, numberOfBoids >> > (dev_positions, dev_velocities, boidRadius);
	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

__global__  void updatePositionsWithVelocities(float2 *positions, float2 *velocities, float boidradius)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if (boidIndex < numberOfBoids)
	{
		float2 alignmentVector = alignment(boidIndex, positions, velocities, boidradius);
		float2 cohesionVector = cohesion(boidIndex, positions, velocities, boidradius);
		float2 separationVector = separation(boidIndex, positions, velocities, boidradius);
		velocities[boidIndex] = calculateBoidVelocity(velocities[boidIndex], alignmentVector,
			cohesionVector, separationVector);
		positions[boidIndex].x += velocities[boidIndex].x;
		positions[boidIndex].y += velocities[boidIndex].y;
	}
}

void loadPositionOffsetOnVBO()
{
	glBindBuffer(GL_ARRAY_BUFFER, translationsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfBoids, &positions[0], GL_DYNAMIC_DRAW);
}

void endApplication()
{
	freeCUDADataStructures();
	printf("%s completed, returned %s\n", windowTitle, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void freeCUDADataStructures()
{
	cudaFree(dev_positions);
	cudaFree(dev_velocities);
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

int randomMinusOneOrOneInt()
{
	return (int)rand() % 2 * 2 - 1;;
}

float randomMinusOneOrOneFloat()
{
	return (float)(rand() % 101) / 100 * 2 - 1;;
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
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

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
		sendFlockToMouseClick(x, y);
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void sendFlockToMouseClick(int x, int y)
{
	float2 destination = mouseToWorldCoordinates(x, y);
	setFlockDestination(destination);
}

float2 mouseToWorldCoordinates(int x, int y)
{
	float fX = (float)x / window_width;
	float fY = (float)y / window_width;
	fX = fX * 2 - 1;
	fY = -fY * 2 + 1;
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
