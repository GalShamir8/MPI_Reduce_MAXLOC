#include "heron.h"

double heron(int n, double epsilon, int *counter)
{
  double x = 10.0; // first approximation (we could do better ...)

  *counter = 0;
  double next_x = x;
  do
  {
    x = next_x;
    next_x = (x + n / x) / 2;
    (*counter)++; //  count number of iterations
  } while (fabs(next_x - x) > epsilon);

  return next_x;
}