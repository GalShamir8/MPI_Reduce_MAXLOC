#include <stdio.h>
#include <math.h>
/*
  return n squared root within epsilon tolerance
  store number of iteration took to find n squared root in counter pointer
*/
double heron(int n, double epsilon, int *counter);