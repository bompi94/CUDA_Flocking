#pragma once

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include <vector_functions.h>

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