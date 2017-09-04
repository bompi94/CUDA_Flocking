#include "CudaFlocking.h"
#include "DeviceFunctions.h"
#include "Helper.h"

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__  void computeFlocking(float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int* cellHead, int* cellNext, int* neighbours, float2* temp, int* boidXCellsIDs, int streamNumber); 
__global__ void setupCells(float2 *positions, int* cellHead, int* cellNext, Cell* cells, int numberOfCells, 
	int* boidXCellsIDs, int* neighbours, int streamNumber);
__global__ void makeMovement(float2* positions, float2* velocities,
	int*  cellHead, int* cellNext, Cell* cells, int numberOfCells, float2* temp, int* boidXCellsIds, int  streamNumber);
__device__ void computeAllForBoid(int boidIndex, int boidY, float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int numberOfCells, int* cellHead, int* cellNext, int* neighbours, float2* temp, int* boidXCellsIDs);
__device__ int GetCellId(int myCellID, int* neighbours, Cell* cells, float2 pos, int numberOfCells); 
__device__ void screenOverflow(float2 *positions, int boidIndex); 


CUDAFlocking::CUDAFlocking(){}

void CUDAFlocking::init()
{
	for (size_t i = 0; i < numStreams; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	preparePositionsAndVelocitiesArray();
	prepareObstacles();
	prepareCells();
}

void CUDAFlocking::calculateBoidsPositions()
{
	int threadsPerBlock = 512;

	int numberOfThreadsNeeded = numberOfBoids / boidPerThread;

	dim3 grid(numberOfBoids / threadsPerBlock + 1, 1);
	dim3 computeGrid((numberOfThreadsNeeded / threadsPerBlock + 1), 1);

	setupCells << <grid, dim3(threadsPerBlock, 1) >> >
		(dev_positions, dev_cellHead, dev_cellNext, dev_cells, numberOfCells, dev_boidXCellsIDs, dev_neighbours, 0);

	for (size_t i = 0; i < numStreams; i++)
	{
		computeFlocking << <computeGrid, dim3(threadsPerBlock, 1), 0, streams[i] >> >
			(dev_positions, dev_velocities, boidRadius, dev_obstacleCenters, dev_obstacleRadii, dev_cells,
				dev_cellHead, dev_cellNext, dev_neighbours, dev_temp, dev_boidXCellsIDs, i);
	}

	makeMovement << <grid, dim3(threadsPerBlock, 1) >> >
		(dev_positions, dev_velocities, dev_cellHead, dev_cellNext, dev_cells, numberOfCells, dev_temp, dev_boidXCellsIDs, 0);

	cudaMemcpy(positions, dev_positions, numberOfBoids * sizeof(float2), cudaMemcpyDeviceToHost);
}

float2* CUDAFlocking::getPositions()
{
	return positions;
}

float2* CUDAFlocking::getObstacleCenters()
{
	return obstacleCenters; 
}

float* CUDAFlocking::getObstacleRadii()
{
	return obstacleRadii; 
}

__global__  void computeFlocking(float2 *positions,
	float2 *velocities, float boidradius, float2 *obstacleCenters, float *obstacleRadii,
	Cell* cells, int* cellHead, int* cellNext, int* neighbours, float2* temp, int* boidXCellsIDs, int streamNumber)
{
	unsigned int boidIndex = (blockIdx.x*blockDim.x + threadIdx.x) * CUDAFlocking::boidPerThread;

	for (size_t i = 0; i < CUDAFlocking::boidPerThread; i++)
	{
		int index = boidIndex + i;
		if (index < numberOfBoids)
		{
			computeAllForBoid(index, streamNumber, positions, velocities, boidradius, obstacleCenters, obstacleRadii,
				cells, numberOfCells, cellHead, cellNext, neighbours, temp, boidXCellsIDs);
		}
	}
}

__global__ void setupCells(float2 *positions, int* cellHead, int* cellNext, Cell* cells, int numberOfCells, int* boidXCellsIDs, int* neighbours, int streamNumber)
{
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x + streamNumber*numberOfBoids / CUDAFlocking::numStreams;
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
	unsigned int boidIndex = blockIdx.x*blockDim.x + threadIdx.x + streamNumber*numberOfBoids / CUDAFlocking::numStreams;
	const float boidSpeed = 0.002f;
	if (boidIndex < numberOfBoids)
	{
		int cellID = boidXCellsIds[boidIndex];
		int base = boidIndex * 4;

		float2 boidVelocity = calculateBoidVelocity(velocities[boidIndex],
			temp[base + 0], temp[base + 1], temp[base + 2], temp[base + 3]);
		boidVelocity = normalizeVector(boidVelocity);
		boidVelocity = vectorMultiplication(boidVelocity, boidSpeed);

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
	else if (cells[myCellID].IsPositionInCell(pos))
		return cells[myCellID].id;

	else
		//changing cell, search the next in the neighbours of my cell
		for (int i = 0; i < 8; i++)
		{
			int base = myCellID * 8;
			int currentNeighbourIndex = neighbours[base + i];
			Cell currentNeighbour = cells[currentNeighbourIndex];
			if (currentNeighbour.IsPositionInCell(pos))
				return currentNeighbour.id;
		}
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

void CUDAFlocking::preparePositionsAndVelocitiesArray()
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

void CUDAFlocking::prepareBoidCUDADataStructures()
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

void CUDAFlocking::prepareObstacles()
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

void CUDAFlocking::prepareObstaclesCUDADataStructures()
{
	cudaMalloc((void**)&dev_obstacleCenters, numberOfObstacles * sizeof(float2));
	cudaMemcpy(dev_obstacleCenters, obstacleCenters, numberOfObstacles * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_obstacleRadii, numberOfObstacles * sizeof(float));
	cudaMemcpy(dev_obstacleRadii, obstacleRadii, numberOfObstacles * sizeof(float), cudaMemcpyHostToDevice);
	printf("prepared obstacles in CUDA\n");
}

void CUDAFlocking::prepareCells()
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

void CUDAFlocking::prepareCellsCUDADataStructures()
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


void CUDAFlocking::freeCUDADataStructures()
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

void CUDAFlocking::setFlockDestination(float2 destination)
{
	for (int i = 0; i < numberOfBoids; i++)
	{
		velocities[i].x += destination.x - positions[i].x;
		velocities[i].y += destination.y - positions[i].y;
		velocities[i] = normalizeVector(velocities[i]);
	}
	cudaMemcpy(dev_velocities, velocities, numberOfBoids * sizeof(float2), cudaMemcpyHostToDevice);
}
