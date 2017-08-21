#include "CudaFlocking.h"
#include "Graphics.h"

Graphics graphics; 

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

	graphics.initialize(&argc, argv);

	registerGlutCallbacks();
	preparePositionsAndVelocitiesArray();
	prepareObstacles(); 
	createVBO(&vbo);
	prepareCUDADataStructures();
}

void registerGlutCallbacks()
{
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
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

void prepareObstacles()
{
	for (int i = 0; i < numberOfObstacles; i++)
	{
		obstacleCenters[i] = make_float2(randomMinusOneOrOneFloat()/2, randomMinusOneOrOneFloat()/2);
		obstacleRadii[i] = obstacleRadius;
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(boidVertices), &boidVertices[0], GL_DYNAMIC_DRAW);

	//loading positions of vertices
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	//loading colors
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
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
	prepareBoidCUDADataStructures(); 
	prepareObstaclesCUDADataStructures(); 
}

void prepareBoidCUDADataStructures()
{
	cudaMalloc((void**)&dev_positions, numberOfBoids * sizeof(float2));
	cudaMemcpy(dev_positions, positions, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_velocities, numberOfBoids * sizeof(float2));
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
}

void prepareObstaclesCUDADataStructures()
{
	cudaMalloc((void**)&dev_obstacleCenters, numberOfObstacles * sizeof(float2));
	cudaMemcpy(dev_obstacleCenters, obstacleCenters, numberOfObstacles * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_obstacleRadii, numberOfObstacles * sizeof(float));
	cudaMemcpy(dev_obstacleRadii, obstacleRadii, numberOfObstacles * sizeof(float), cudaMemcpyHostToDevice);
}

void display()
{
	sdkStartTimer(&timer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(VAO);
	drawObstacles();
	calculateBoidsPositions();
	drawBoids(); 
	glutSwapBuffers();
	sdkStopTimer(&timer);
	computeFPS();
}

void drawObstacles()
{
	for (int i = 0; i < numberOfObstacles; i++)
	{
		float2 center = obstacleCenters[i];
		float radius = obstacleRadii[i]; 
		graphics.drawCircle(center, radius, 100);
	}
}

void calculateBoidsPositions()
{
	updatePositionsWithVelocities << <numberOfBoids / 256 + 1, 256 >> > (dev_positions, dev_velocities, boidRadius, dev_obstacleCenters, dev_obstacleRadii);
	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

void drawBoids()
{
	loadPositionOnVBO();
	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, numberOfBoids);
}

__global__  void updatePositionsWithVelocities(float2 *positions, float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii)
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

void loadPositionOnVBO()
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
	cudaFree(dev_obstacleCenters); 
	cudaFree(dev_obstacleRadii); 
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
		velocities[i].x += destination.x - positions[i].x;
		velocities[i].y += destination.y - positions[i].y;
		velocities[i] = normalizeVector(velocities[i]);
	}
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
}
