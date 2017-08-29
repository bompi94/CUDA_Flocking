#pragma once

#include <vector_functions.h>

class Cell {

public:
	int id; 
	float2 topLeftCorner;
	float side;

	__device__ __host__ Cell() //default constructor
	{
		topLeftCorner = make_float2(0, 0);
		side = 0;
	}

	__device__ __host__ Cell(float2 corner, float s, int i)
	{
		topLeftCorner = corner;
		side = s;
		id = i; 
	};
};