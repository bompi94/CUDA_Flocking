#include "Helper.h"
#include <random>

int Helper::randomMinusOneOrOneInt()
{
	return (int)rand() % 2 * 2 - 1;;
}

float Helper::randomMinusOneOrOneFloat()
{
	return (float)(rand() % 101) / 100 * 2 - 1;
}
