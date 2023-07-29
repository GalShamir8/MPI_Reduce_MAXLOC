#include "main.h"

const int MIN_NUM_OF_PROC = 2;
const int DEFAULT_INPUT = 100;
const int ROOT_PROCESS_RANK = 0;
const double DEFAULT_EPSILON = 0.01;

/*
  Example
*/
// int main(int argc, char **argv)
// {
//   int rank;
//   int numOfProc;
//   init(&argc, &argv, &numOfProc, &rank);
//   printf("hello from %d out of %d processes\n", rank, numOfProc);
// }

void init(int *argc, char **argv, int *numOfProc, int *rank, int *input, double *epsilon)
{
  mpiInit(argc, argv, numOfProc, rank);
  setInputs(input, epsilon, argc, argv);
  if (*numOfProc < MIN_NUM_OF_PROC)
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void mpiInit(int *argc, char **argv, int *numOfProc, int *rank)
{
  MPI_Init(argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, numOfProc);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void setInputs(int *input, double *epsilon, int *argc, char **argv)
{
  if (*argc < 3)
  {
    *input = DEFAULT_INPUT;
    *epsilon = DEFAULT_EPSILON;
    return;
  }

  *input = atoi(argv[1]);
  *epsilon = atof(argv[2]);
  if (*epsilon > 1 || *epsilon < 0) *epsilon = DEFAULT_EPSILON;
}

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
