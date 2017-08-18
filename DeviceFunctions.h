#pragma once

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include <vector_functions.h>

#include "Boid.h"


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

	return make_float2(vector.x / scalar, vector.y / scalar);
}

__device__ __host__ float2 normalizeVector(float2 vector)
{
	///normalization of velocity of the boid
	float length = sqrtf((vector.x * vector.x) + (vector.y * vector.y));
	vector.x /= length;
	vector.y /= length;

	return vector; 
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

__device__ float2 calculateBoidVelocity(float2 velocityOfTheBoid, float2 alignmentVector, float2 cohesionVector, float2 separationVector)
{
	float alignmentWeight, cohesionWeight, separationWeight;
	alignmentWeight = 4;
	cohesionWeight = 4;
	separationWeight = 4;
	float boidSpeed = 0.01;
	velocityOfTheBoid.x += alignmentVector.x * alignmentWeight
		+ cohesionVector.x * cohesionWeight
		+ separationVector.x * separationWeight;
	velocityOfTheBoid.y += alignmentVector.y * alignmentWeight
		+ cohesionVector.y * cohesionWeight
		+ separationVector.y * separationWeight;
	velocityOfTheBoid = normalizeVector(velocityOfTheBoid);
	velocityOfTheBoid = vectorMultiplication(velocityOfTheBoid, boidSpeed);
	return velocityOfTheBoid;
}
