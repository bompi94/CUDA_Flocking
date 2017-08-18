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

__global__  void updatePositionsWithVelocities(float2 *positions, float2 *velocities, float boidradius)
{
	unsigned int threadX = blockIdx.x*blockDim.x + threadIdx.x;

	float alignmentWeight, cohesionWeight, separationWeight; 
	alignmentWeight = 4; 
	cohesionWeight = 4;
	separationWeight = 4; 
	float boidSpeed = 0.01;
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

void launchFlockingKernel()
{
	updatePositionsWithVelocities << <1, 512 >> > (dev_positions, dev_velocities, boidRadius);
	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

void preparePositionsAndVelocitiesArray()
{
	for (int i = 0; i < numberOfBoids; i++)
	{
		int a = randomMinusOneOrOne();
		int b = randomMinusOneOrOne();
		velocities[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
		velocities[i] = normalizeVector(velocities[i]);
		positions[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
	}
}

void prepareCUDADataStructures()
{
	cudaMalloc((void**)&dev_positions, numberOfBoids * sizeof(float2));
	cudaMemcpy(dev_positions, positions, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_velocities, numberOfBoids * sizeof(float2));
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
}

void freeCUDADataStructures()
{
	cudaFree(dev_positions);
	cudaFree(dev_velocities);
}

void endApplication()
{
	freeCUDADataStructures();
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

int randomMinusOneOrOne()
{
	return rand() % 2 * 2 - 1;;
}

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

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void loadPositionOffsetOnVBO()
{
	glBindBuffer(GL_ARRAY_BUFFER, translationsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfBoids, &positions[0], GL_DYNAMIC_DRAW);
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

void sendFlockToMouseClick(int x, int y)
{
	float2 destination = mouseToWorldCoordinates(x, y);
	setFlockDestination(destination);
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
		sendFlockToMouseClick(x,y); 
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}