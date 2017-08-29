#pragma once
#ifndef _CELL_H
#define _CELL_H

#include <vector_functions.h>
#include <stdlib.h>

unsigned int numberOfCells = 5;

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

	//assuming the grid has at least 8 cells
	__device__ __host__  int* getAdjacentCells()
	{
		int n = numberOfCells + 1;
		int* result = (int*)malloc(sizeof(int) * 8);

		result[0] = specialSum(id, -n);
		result[1] = specialSum(id, -n + 1);
		result[2] = specialSum(id, -n + 2);

		result[3] = specialSum(id, -1);
		result[4] = specialSum(id, 1);

		result[5] = specialSum(id, n - 2);
		result[6] = specialSum(id, n - 1);
		result[7] = specialSum(id, n);

		return result;
	}

private:
	bool onSameRow(int a, int b)
	{
		return a / numberOfCells == b / numberOfCells;
	}

	int specialSum(int id, int toSum)
	{
		int s = id + toSum;
		if (s == id - 1) {
			if (!onSameRow(s, id) || s == -1) {
				s += numberOfCells;
			}
		}

		else if (s == id + 1) {
			if (!onSameRow(s, id))
				s -= numberOfCells;
		}

		else if (s < id - 1 || s == 0) {
			if (s < 0) {
				s += numberOfCells * numberOfCells;
				if (s < 0) //copre il caso dello 0 
					s += 1;
			}

			if (s == 0) {
				s += numberOfCells * numberOfCells - numberOfCells;
			}
			if (onSameRow(s, id))
				s -= 1; 
		}

		else if (s > id + 1)
		{
			if (s > numberOfCells * numberOfCells) {
				s -= numberOfCells * numberOfCells;
				if (s > numberOfCells)
					s -= 1; //copre il caso del bottom right
				if (s < 0) // copre il caso del bottom left
					s += 1;
			}

			if (onSameRow(s, id))
				s += numberOfCells; 
		}

		return s;
	}
};

#endif