#include "Cell.h"
#include <stdio.h>

bool IsPositionInCell(float2 pos, Cell cell); 


bool IsPositionInCell(float2 position, Cell c)
{
	float x0, y0, x1, y1; 
	x0 = c.topLeftCorner.x; 
	y0 = c.topLeftCorner.y; 
	x1 = x0 + c.side;
	y1 = y0 - c.side; 
	bool correctX = position.x >= x0 && position.x <= x1; 
	bool correctY = position.y <= y0 && position.y >= y1; 
	return correctX && correctY; 
}




