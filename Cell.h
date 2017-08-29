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

	__device__ bool IsPositionInCell(float2 position)
	{
		float x0, y0, x1, y1;
		x0 = topLeftCorner.x;
		y0 = topLeftCorner.y;
		x1 = x0 + side;
		y1 = y0 - side; 
		bool correctX = position.x >= x0 && position.x <= x1;
		bool correctY = position.y <= y0 && position.y >= y1;
		return correctX && correctY;
	}
};