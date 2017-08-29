#include "Grid.h"
#include <stdio.h>

bool PositionInCell(float2 pos, Cell cell); 

Grid::Grid(int n )
{
	nCells = n; 
	float x = -1; 
	float y = 1; 
	float side = (float) 2 / nCells;
	for (int i = 0; i<nCells; i++)
	{
		for (int j = 0; j < nCells; j++)
		{
			float2 corner = make_float2(x, y); 
			Cell c(corner,side); 
			x += side; 
			std::pair<Cell, std::list<int>> pair(c, std::list<int>()); 
			cellToBoids.insert(pair); 
		}
		y -= side; 
		x = -1; 
	}
}

Cell Grid::FindCellFromPosition(float2 pos)
{
	Cell c;
	for (std::map<Cell, std::list<int>>::iterator it = cellToBoids.begin(); it != cellToBoids.end(); ++it) {
		if (PositionInCell(pos, it->first))
		{
			return it->first; 
		}
	}
}

bool PositionInCell(float2 position, Cell c)
{
	return true; 
}

void Grid::Register(float2 pos, int boidIndex)
{
	Cell c = FindCellFromPosition(pos);
	cellToBoids[c].push_back(boidIndex); 
	boidToCell.insert(std::pair<int, Cell>(boidIndex, c)); 
}

Cell Grid::GetCellOf(int boidIndex)
{
	return boidToCell[boidIndex];
}

std::list<int> Grid::GetNeighbors(int boidIndex)
{
	return cellToBoids[GetCellOf(boidIndex)]; //per ora ritorna tutti i boid nella stessa cella, poi andranno aggiunte le vicine
}

void Grid::EmptyCells()
{
	for (std::map<Cell, std::list<int>>::iterator it = cellToBoids.begin(); it != cellToBoids.end(); ++it) {
		it->second.clear();
	}
}

void Grid::PrintCellToBoids()
{
	for (std::map<Cell, std::list<int>>::iterator it = cellToBoids.begin(); it != cellToBoids.end(); ++it) {
		printf("tlc = %f %f \n", it->first.topLeftCorner.x, it->first.topLeftCorner.y);
		printf("boids "); 
		for (std::list<int>::iterator i = it->second.begin(); i != it->second.end(); ++i)
		{
			printf("%d ", *i); 
		}
	}
}



