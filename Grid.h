#pragma once
#include <vector_functions.h>
#include <map>
#include <list>

class Cell {

public:
	Cell(float2 corner, float s)
	{
		topLeftCorner = corner;
		side = s;
	};
	Cell() //default constructor
	{
		topLeftCorner = make_float2(0, 0); 
		side = 0; 
	}
	float2 topLeftCorner;
	float side;

};

namespace std
{
	template<> struct less<Cell>
	{
		bool operator() (const Cell& lhs, const Cell& rhs) const
		{
			return lhs.topLeftCorner.x < rhs.topLeftCorner.x || lhs.topLeftCorner.y > rhs.topLeftCorner.y;
		}
	};
}

class Grid {
public:
	Grid(int nCells);
	void Register(float2 pos, int boidIndex);
	std::list<int> GetNeighbors(int boidIndex);
	void EmptyCells(); 
protected:
	unsigned int nCells;
	std::map<Cell, std::list<int>> cellToBoids;
	std::map<int, Cell> boidToCell;

	Cell GetCellOf(int boidIndex);
	void PrintCellToBoids(); 
	Cell FindCellFromPosition(float2 position); 
};