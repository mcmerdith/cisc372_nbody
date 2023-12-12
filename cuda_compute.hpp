#pragma once
#include "compute.hpp"
#include "vector.hpp"

#define DIM_SIZE(dim) (dim.x * dim.y * dim.z)

#define CEIL_DIVIDE(x, y) ((x + y - 1) / y)

#define SQUARE(x) (x * x)

#define INDEX(row, col, width) ((row * width) + col)
#define ENT_INDEX(row, col) INDEX(row, col, NUMENTITIES)

void compute_prepare();
void compute_complete();