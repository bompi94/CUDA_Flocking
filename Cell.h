#pragma once
#ifndef _CELL_H
#define _CELL_H

#include <vector_functions.h>
#include <stdlib.h>

unsigned int numberOfCells = 10;

class Cell {
public:
	int id;
	float2 topLeftCorner;
	float side;

	__device__ __host__ Cell() //default constructor
	{
		topLeftCorner = make_float2(0, 0);
		side = 0;
		numberOfCells = numberOfCells;
	}

	__device__ __host__ Cell(float2 corner, float s, int i)
	{
		topLeftCorner = corner;
		side = s;
		id = i;
		numberOfCells = numberOfCells; 
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
	static int* getAdjacentCells(int id)
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
	static bool onSameRow(int a, int b)
	{
		return a / numberOfCells == b / numberOfCells;
	}

	 static int specialSum(int id, int toSum)
	{
		int s = id + toSum;

		//handles left case
		if (s == id - 1) {
			if (!onSameRow(s, id) || s == -1) {
				s += numberOfCells;
			}
		}

		//handles right case
		else if (s == id + 1) {
			if (!onSameRow(s, id) || s == numberOfCells*numberOfCells)
				s -= numberOfCells;
		}

		//handles down cases
		else if (s > id + 1) {

			//exactly down
			if (s == id + numberOfCells) {
				//overflow down
				if (s >= numberOfCells * numberOfCells)
				{
					s -= numberOfCells * numberOfCells;
				}
			}

			//down right
			else if (s == id + numberOfCells + 1)
			{


				//overflow
				if (s >= numberOfCells*numberOfCells) {
					s -= numberOfCells*numberOfCells;
					//handles bottom right case 
					if (s / numberOfCells > 0) {
						s -= numberOfCells;
					}
				}

				//two lines down
				else if (s / numberOfCells - id / numberOfCells == 2)
				{
					s -= numberOfCells;
				}
			}

			//down left
			else if (s == id + numberOfCells - 1) {
				//overflow
				if (s >= numberOfCells * numberOfCells) {
					s -= numberOfCells*numberOfCells;
					//handles bottom left case
					if (s == -1)
						s += numberOfCells;
				}
				//same line
				else if (onSameRow(s, id)) {
					s += numberOfCells;
					if (s >= numberOfCells*numberOfCells) {
						s -= numberOfCells*numberOfCells; 
					}
				}


			}
		}

		//handles up cases
		else if (s < id - 1) {
			//exactly up
			if (s == id - numberOfCells) {
				if (s < 0) {
					s += numberOfCells*numberOfCells;
				}
			}

			//up right
			else if (s == id - numberOfCells + 1) {

				//overflow
				if (s < 0) {
					s += numberOfCells * numberOfCells;
					//top right
					if (s >= numberOfCells * numberOfCells) {
						s -= numberOfCells;
					}
				}
				//same row
				else if (onSameRow(s, id)) {
					s -= numberOfCells;
					if (s < 0) {
						s += numberOfCells * numberOfCells;
					}
				}

			}

			//up left
			else if (s == id - numberOfCells - 1) {
				if (s == -1) {
					s += numberOfCells;
				}
				//overflow
				else if (s < 0) {
					s += numberOfCells*numberOfCells;
					if (s < numberOfCells*numberOfCells - numberOfCells)
						s += numberOfCells;
				}
				//two lines up
				else if (s / numberOfCells - id / numberOfCells == -2) {
					s += numberOfCells;
				}
			}


		}

		return s;
	}
};

#endif