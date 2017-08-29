#include "CudaFlocking.h"

#include "Graphics.h"
#include "Helper.h"

Graphics graphics;
Cell* cells;
Cell* dev_cells;

unsigned int numberOfCells = 3;

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
	printf("%s starting...\n", graphics.windowTitle);
	sdkCreateTimer(&timer);
	graphics.initialize(&argc, argv);
	registerGlutCallbacks();
	preparePositionsAndVelocitiesArray();
	prepareObstacles();
	prepareCells(); 
	prepareGraphicsToRenderBoids(&vbo);
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
		int a = Helper::randomMinusOneOrOneInt();
		int b = Helper::randomMinusOneOrOneInt();
		velocities[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
		velocities[i] = normalizeVector(velocities[i]);
		positions[i] = make_float2(Helper::randomMinusOneOrOneFloat(), Helper::randomMinusOneOrOneFloat());
	}
}

void prepareCells()
{
	cells = (Cell*)malloc(sizeof(Cell) * numberOfCells * numberOfCells); 

	float side = (float)2.1 / numberOfCells; 
	float x = -1.05; 
	float y = 1.05; 

	unsigned int id = 0; 

	for (int i = 0; i < numberOfCells; i++)
	{
		x = -1.05; 
		for (int j = 0; j < numberOfCells; j++) 
		{

			Cell c(make_float2(x,y), side, id);

			cells[id] = c;
			id++;
			x += side; 
		}
		y -= side; 
	}
}

void prepareObstacles()
{
	for (int i = 0; i < numberOfObstacles; i++)
	{
		obstacleCenters[i] = make_float2(Helper::randomMinusOneOrOneFloat() / 2, Helper::randomMinusOneOrOneFloat() / 2);
		obstacleRadii[i] = obstacleRadius;
	}
}

void prepareGraphicsToRenderBoids(GLuint *vbo)
{
	graphics.createGLStructures(vbo, &VAO);
	graphics.saveBoidsRenderingData(vbo, boidVertices, numberOfBoids);
	graphics.loadBoidsVertices(vbo);
	graphics.loadBoidsColor(vbo);
	graphics.loadBoidsPosition(vbo, &translationsVBO, positions, numberOfBoids);
	graphics.allowInstancing();
}

void prepareCUDADataStructures()
{
	prepareBoidCUDADataStructures();
	prepareObstaclesCUDADataStructures();
	prepareCellsCUDADataStructures(); 
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

void prepareCellsCUDADataStructures()
{
	cudaMalloc((void**)&dev_cells, numberOfCells * numberOfCells * sizeof(Cell));
	cudaMemcpy(dev_cells, cells, numberOfCells * numberOfCells * sizeof(Cell), cudaMemcpyHostToDevice);
}

void startOfFrame()
{
	sdkStartTimer(&timer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(VAO);
}

void endOfFrame()
{
	glutSwapBuffers();
	sdkStopTimer(&timer);
	computeFPS();
}

void display()
{
	startOfFrame();
	graphics.drawObstacles(numberOfObstacles, obstacleCenters, obstacleRadii);
	calculateBoidsPositions();
	graphics.drawBoids(numberOfBoids, &translationsVBO, positions);
	endOfFrame();
}

void callKernel()
{
	int threadsPerBlock = 32;
	updatePositionsWithVelocities1 << <numberOfBoids / threadsPerBlock + 1, threadsPerBlock >> >
		(dev_positions, dev_velocities, boidRadius, dev_obstacleCenters, dev_obstacleRadii, dev_cells, numberOfCells);
}

void calculateBoidsPositions()
{
	callKernel();
	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

void endApplication()
{
	freeCUDADataStructures();
	free(cells); 
	printf("%s completed, returned %s\n", graphics.windowTitle, (g_TotalErrors == 0) ? "OK" : "ERROR!");
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
