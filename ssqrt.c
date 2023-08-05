# include "ssqrt.h"

int main(int argc, char **argv)
{
  int rank, numOfProc, input;
  double epsilon;

  initSerial(&argc, argv, &numOfProc, &rank, &input, &epsilon);
  double start = MPI_Wtime();
  process(input, epsilon);
  printf("Runtime: %lf\n",MPI_Wtime() - start);
  MPI_Finalize();
  return 0;
}

void process(int input, double epsilon)
{
  int max = -1;
  int numWithMax = 1;
  for(int n = 1; n <= input; n++)
  {
    int numOfIterations;
    heron(n, epsilon, &numOfIterations);    
    if (numOfIterations > max)
    {      
      max = numOfIterations;
      numWithMax = n;
    }
  }  
  printf(
      "[sequential] number requiring max number of iterations: %d. (number of iterations: %d)\n",
      numWithMax,
      max
  );
}