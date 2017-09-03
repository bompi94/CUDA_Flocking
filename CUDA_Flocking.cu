#include "CudaFlocking.h"


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

int main(int argc, char **argv)
{
	printf("quinto (stream) approccio boids -> %d\n", numberOfBoids);
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

	//prepare streams

	for (size_t i = 0; i < numStreams; i++)
	{
		cudaStreamCreate(&streams[i]);
	}

	preparePositionsAndVelocitiesArray();
	prepareObstacles();
	prepareCells();
	prepareGraphicsToRenderBoids(&vbo);
}

void registerGlutCallbacks()
{
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutCloseFunc(cleanup);
	printf("registered glut callbacks\n");
}

void preparePositionsAndVelocitiesArray()
{
	cudaMallocHost((void**)&positions, numberOfBoids * sizeof(float2));
	cudaMallocHost((void**)&velocities, numberOfBoids * sizeof(float2));

	for (int i = 0; i < numberOfBoids; i++)
	{
		int a = Helper::randomMinusOneOrOneInt();
		int b = Helper::randomMinusOneOrOneInt();
		velocities[i] = make_float2(a*(float)(rand() % 10) / 50, b*(float)(rand() % 10) / 50);
		velocities[i] = normalizeVector(velocities[i]);
		positions[i] = make_float2(Helper::randomMinusOneOrOneFloat(), Helper::randomMinusOneOrOneFloat());
	}
	printf("prepared positions and velocities\n");
	prepareBoidCUDADataStructures();
}

void prepareBoidCUDADataStructures()
{
	cudaMalloc((void**)&dev_positions, numberOfBoids * sizeof(float2));
	for (size_t i = 0; i < numStreams; i++)
	{
		cudaMemcpyAsync(&dev_positions[i*offset], &positions[i*offset], offset * sizeof(float2), cudaMemcpyHostToDevice, streams[i]);
	}


	cudaMalloc((void**)&dev_velocities, numberOfBoids * sizeof(float2));
	for (size_t i = 0; i < numStreams; i++)
	{
		cudaMemcpyAsync(&dev_velocities[i*offset], &velocities[i*offset], offset * sizeof(float2), cudaMemcpyHostToDevice, streams[i]);
	}

	float2* temp;
	cudaMallocHost((void**)&temp, 4 * numberOfBoids * sizeof(float2));
	for (int i = 0; i < 4 * numberOfBoids; i++)
	{
		temp[i] = make_float2(0, 0);
	}

	cudaMalloc((void**)&dev_temp, sizeof(float2) * 4 * numberOfBoids);
	for (size_t i = 0; i < numStreams; i++)
	{
		cudaMemcpyAsync(dev_temp, temp, sizeof(float2) * 4 * offset, cudaMemcpyHostToDevice, streams[i]);
	}
	printf("prepared positions and velocities in CUDA\n");
}

void prepareObstacles()
{
	cudaMallocHost((void**)&obstacleCenters, numberOfObstacles * sizeof(float2));
	cudaMallocHost((void**)&obstacleRadii, numberOfObstacles * sizeof(float));

	for (int i = 0; i < numberOfObstacles; i++)
	{
		obstacleCenters[i] = make_float2(Helper::randomMinusOneOrOneFloat() / 2, Helper::randomMinusOneOrOneFloat() / 2);
		obstacleRadii[i] = obstacleRadius;
	}
	printf("prepared obstacles\n");
	prepareObstaclesCUDADataStructures();
}

void prepareObstaclesCUDADataStructures()
{
	cudaMalloc((void**)&dev_obstacleCenters, numberOfObstacles * sizeof(float2));
	cudaMemcpy(dev_obstacleCenters, obstacleCenters, numberOfObstacles * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_obstacleRadii, numberOfObstacles * sizeof(float));
	cudaMemcpy(dev_obstacleRadii, obstacleRadii, numberOfObstacles * sizeof(float), cudaMemcpyHostToDevice);
	printf("prepared obstacles in CUDA\n");
}

void prepareCells()
{
	cudaMallocHost((void**)&cells, sizeof(Cell) * numberOfCells * numberOfCells);

	float side = (float)2 / numberOfCells;
	float x = -1;
	float y = 1;
	unsigned int id = 0;
	for (int i = 0; i < numberOfCells; i++)
	{
		x = -1;
		for (int j = 0; j < numberOfCells; j++)
		{

			Cell c(make_float2(x, y), side, id);

			cells[id] = c;
			id++;
			x += side;
		}
		y -= side;
	}

	cudaMallocHost((void**)&cellHead, sizeof(int)*numberOfCells * numberOfCells);
	cudaMallocHost((void**)&cellNext, sizeof(int)*numberOfBoids);

	//-1 represents an invalid boid index for the cell, it means that the cell is empty	
	for (int i = 0; i < numberOfCells * numberOfCells; i++)
	{
		cellHead[i] = -1;
	}

	//-1 represents the end of the chain of references for the cell, no more boids in the cell
	for (int i = 0; i < numberOfBoids; i++)
	{
		cellNext[i] = -1;
	}

	cudaMallocHost((void**)&neighbours, sizeof(int*) * numberOfCells * numberOfCells);
	for (int i = 0; i < numberOfCells * numberOfCells; i++) {
		int * neighbourCells = Cell::getAdjacentCells(i);
		neighbours[i] = neighbourCells;
	}
	printf("prepared cells\n");
	prepareCellsCUDADataStructures();
}

void prepareCellsCUDADataStructures()
{
	cudaMalloc((void**)&dev_cells, numberOfCells * numberOfCells * sizeof(Cell));
	cudaMemcpy(dev_cells, cells, numberOfCells * numberOfCells * sizeof(Cell), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_cellHead, numberOfCells * numberOfCells * sizeof(int));
	cudaMemcpy(dev_cellHead, cellHead, numberOfCells * numberOfCells * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_cellNext, numberOfBoids * sizeof(int));
	cudaMemcpy(dev_cellNext, cellNext, numberOfBoids * sizeof(int), cudaMemcpyHostToDevice);


	cudaMalloc((void**)&dev_neighbours, (numberOfCells * numberOfCells * 8) * sizeof(int));
	int* linearizedNeighbours;
	cudaMallocHost((void**)&linearizedNeighbours, (numberOfCells * numberOfCells * 8) * sizeof(int));
	int cont = 0;
	for (int i = 0; i < numberOfCells*numberOfCells; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			linearizedNeighbours[cont] = neighbours[i][j];
			cont++;
		}
	}
	cudaMemcpy(dev_neighbours, linearizedNeighbours, (numberOfCells * numberOfCells * 8) * sizeof(int), cudaMemcpyHostToDevice);

	int* bxci;
	cudaMallocHost((void**)&bxci, numberOfBoids * sizeof(int));
	for (int i = 0; i < numberOfBoids; i++)
	{
		bxci[i] = -1;
	}
	cudaMalloc((void**)&dev_boidXCellsIDs, sizeof(int)*numberOfBoids);
	cudaMemcpy(dev_boidXCellsIDs, bxci, sizeof(int)*numberOfBoids, cudaMemcpyHostToDevice);
	printf("prepared cells in CUDA\n");
}

void prepareGraphicsToRenderBoids(GLuint *vbo)
{
	graphics.createGLStructures(vbo, &VAO);
	printf(" graphics->createdGLstructures\n");

	graphics.saveBoidsRenderingData(vbo, boidVertices, 15);
	printf(" graphics->savedBoidsRenderingData\n");

	graphics.loadBoidsPosition(vbo, &translationsVBO, positions, numberOfBoids);
	printf(" graphics->loadedBoidsPosition\n");

	graphics.allowInstancing();
	printf(" graphics->allowedInstancing\n");

	printf("prepared graphics to rendering\n");
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

void calculateBoidsPositions()
{
	int threadsPerBlock = 512;

	int numberOfThreadsNeeded = numberOfBoids / boidPerThread;

	dim3 grid(numberOfBoids / threadsPerBlock + 1, 1);
	dim3 computeGrid((numberOfThreadsNeeded / threadsPerBlock + 1), 1);
	dim3 lesserGrid(offset / threadsPerBlock + 1, 1);

	setupCells << <grid, dim3(threadsPerBlock, 1) >> >
		(dev_positions, dev_cellHead, dev_cellNext, dev_cells, numberOfCells, dev_boidXCellsIDs, dev_neighbours, 0);

	for (size_t i = 0; i < numStreams; i++)
	{
		computeFlocking << <computeGrid, dim3(threadsPerBlock,1), 0, streams[i] >> >
			(dev_positions, dev_velocities, boidRadius, dev_obstacleCenters, dev_obstacleRadii, dev_cells,
				dev_cellHead, dev_cellNext, dev_neighbours, dev_temp, dev_boidXCellsIDs, i);
	}

	makeMovement << <grid, dim3(threadsPerBlock, 1) >> >
		(dev_positions, dev_velocities, dev_cellHead, dev_cellNext, dev_cells, numberOfCells, dev_temp, dev_boidXCellsIDs, 0);

	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

void endApplication()
{
	//freeCUDADataStructures();
	printf("%s completed, returned %s\n", graphics.windowTitle, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void freeCUDADataStructures()
{
	cudaFree(dev_positions);
	cudaFree(dev_velocities);
	cudaFree(dev_obstacleCenters);
	cudaFree(dev_obstacleRadii);
	cudaFree(dev_temp);
	cudaFree(dev_cells);
	cudaFree(dev_cellHead);
	cudaFree(dev_cellNext);
	cudaFree(dev_neighbours);
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
		glutDestroyWindow(glutGetWindow());
		return;
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
