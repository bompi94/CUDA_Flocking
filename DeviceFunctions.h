#pragma once
#ifndef DEVICEFUNCTIONS_H
#define DEVICEFUNCTIONS_H

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <vector_types.h>
#include <vector_functions.h>
#include "Boid.h"
#include "Obstacle.h"

__device__ Obstacle findMostThreateningObstacle(float2 position, float2 ahead, float2 ahead2, float2* obstacleCenters, float* obstacleRadii);
__device__ bool lineIntersectsCircle(float2 position, float2 ahead, float2 ahead2, float2 obstacleCenter, float obstacleRadius);

float* dev_obstacleRadii;
float2* dev_obstacleCenters;

__device__ __host__ void DebugPrintFloat2(float2 vector)
{
	printf("%f %f \n", vector.x, vector.y);
}

__device__ float distanceBetweenPoints(float2 point1, float2 point2)
{
	return sqrtf(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2));
}

__device__ float2 vectorMultiplication(float2 vector, float scalar)
{
	return make_float2(vector.x*scalar, vector.y*scalar);
}

__device__ float2 vectorDivision(float2 vector, float scalar)
{
	if (scalar != 0)
		return make_float2(vector.x / scalar, vector.y / scalar);
	else
		return vector;
}

__device__ __host__ float2 normalizeVector(float2 vector)
{
	float length = sqrtf((vector.x * vector.x) + (vector.y * vector.y));
	if (length != 0)
	{
		vector.x /= length;
		vector.y /= length;
	}
	return vector;
}





__device__ float2 alignment(int threadX, float2 *positions, float2 *velocities, float boidradius,
	int* cellNext)
{
	float2 alignmentVector = make_float2(0, 0);
	int cont = 0;
	int nextID = cellNext[threadX];

	while (nextID > -1) {
		alignmentVector.x += velocities[nextID].x;
		alignmentVector.y += velocities[nextID].y;
		cont++;
		nextID = cellNext[nextID];
	}

	if (cont != 0) {
		alignmentVector = vectorDivision(alignmentVector, cont);
	}

	alignmentVector = normalizeVector(alignmentVector);
	return alignmentVector;
}

__device__ float2 cohesion(int originalBoid, int firstNeighborBoid, float2 *positions, float2 *velocities, float boidradius, int* cellNext)
{
	float2 cohesionVector = make_float2(0, 0);
	int cont = 0;
	int nextID = cellNext[firstNeighborBoid];

	while (nextID > -1) {
		cohesionVector.x += positions[nextID].x;
		cohesionVector.y += positions[nextID].y;
		cont++;
		nextID = cellNext[nextID];
	}

	if (cont != 0) {
		cohesionVector = vectorDivision(cohesionVector, cont);
		cohesionVector.x -= positions[originalBoid].x;
		cohesionVector.y -= positions[originalBoid].y;
		cohesionVector = normalizeVector(cohesionVector);
	}

	return cohesionVector;
}

__device__ float2 separation(int originalBoid, int neighbourBoid, float2 *positions, float2 *velocities, float boidradius, int* cellNext)
{
	float2 separationVector = make_float2(0, 0);
	int cont = 0;
	int nextID = cellNext[neighbourBoid];

	while (nextID > -1 && nextID<numberOfBoids) {
		separationVector.x += positions[nextID].x - positions[originalBoid].x;
		separationVector.y += positions[nextID].y - positions[originalBoid].y;
		cont++;
		nextID = cellNext[nextID];
	}

	if (cont != 0) {
		separationVector = vectorDivision(separationVector, cont);
		separationVector.x *= -1;
		separationVector.y *= -1;
		separationVector = normalizeVector(separationVector);
	}
	return separationVector;
}





__global__ void DebugPrintNeighbours(int* neighbours, int numberOfCells)
{

	for (int i = 0; i < numberOfCells * numberOfCells * 8; i++)
	{
		if (i % 8 == 0)
			printf("vicini di %d \n", i / 8);
		printf("  %d\n", neighbours[i]);
	}
}

__device__ __host__ float2 vectorSum(float2 vector1, float2 vector2)
{
	return make_float2(vector1.x + vector2.x, vector1.y + vector2.y);
}

__device__ float2 obstacleAvoidance(float2 position, float2 velocity, float2 * obstacleCenters, float* obstacleRadii)
{
	float2 ahead = vectorMultiplication(vectorSum(position, velocity), 0.5);
	float2 ahead2 = vectorMultiplication(vectorSum(position, velocity), 0.5);
	Obstacle o = findMostThreateningObstacle(position, ahead, ahead2, obstacleCenters, obstacleRadii);
	float2 avoidance = make_float2(0, 0);
	if (o.initialized) {
		avoidance.x = ahead.x - o.center.x;
		avoidance.y = ahead.y - o.center.y;
		avoidance = normalizeVector(avoidance);
	}
	else {
		avoidance = vectorMultiplication(avoidance, 0); // nullify the avoidance force
	}
	return avoidance;
}

__device__ Obstacle findMostThreateningObstacle(float2 position, float2 ahead, float2 ahead2, float2* obstacleCenters, float* obstacleRadii)
{
	Obstacle mostThreatening;
	for (int i = 0; i < numberOfObstacles; i++)
	{
		Obstacle temp;
		temp.center = obstacleCenters[i];
		temp.radius = obstacleRadii[i];
		bool collision = lineIntersectsCircle(position, ahead, ahead2, temp.center, temp.radius);
		if (collision && (!mostThreatening.initialized || distanceBetweenPoints(position, temp.center) < distanceBetweenPoints(position, mostThreatening.center)))
		{
			mostThreatening = temp;
			mostThreatening.initialized = true;
		}
	}
	return mostThreatening;
}

__device__ bool lineIntersectsCircle(float2 position, float2 ahead, float2 ahead2, float2 obstacleCenter, float obstacleRadius)
{
	bool aheadInObstacle = distanceBetweenPoints(obstacleCenter, ahead) <= obstacleRadius;
	bool ahead2inObstacle = distanceBetweenPoints(obstacleCenter, ahead2) <= obstacleRadius;
	return aheadInObstacle || ahead2inObstacle;
}

__device__ float2 calculateBoidVelocity(float2 velocityOfTheBoid, float2 alignmentVector, float2 cohesionVector, float2 separationVector, float2 obstacleAvoidanceVector)
{
	float alignmentWeight, cohesionWeight, separationWeight, obstacleAvoidanceWeight;
	alignmentWeight = 100;
	cohesionWeight = 100;
	separationWeight = 100;
	obstacleAvoidanceWeight = 100;

	velocityOfTheBoid.x += alignmentVector.x * alignmentWeight
		+ cohesionVector.x * cohesionWeight
		+ separationVector.x * separationWeight;
	velocityOfTheBoid.y += alignmentVector.y * alignmentWeight
		+ cohesionVector.y * cohesionWeight
		+ separationVector.y * separationWeight;

	velocityOfTheBoid.x += obstacleAvoidanceVector.x * obstacleAvoidanceWeight;
	velocityOfTheBoid.y += obstacleAvoidanceVector.y * obstacleAvoidanceWeight;
	return velocityOfTheBoid;
}

#endif // !DEVICEFUNCTIONS_H