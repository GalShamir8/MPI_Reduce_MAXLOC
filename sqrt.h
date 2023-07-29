# include "main.h"

typedef struct
{
  int input;
  int chunkSize;    
  double epsilon;
} StaticProcessType;

typedef struct
{
  double max;
  int iterations;    
} Result;

void process(int rank, int numOfProc, int input, double epsilon);

void create_type(MPI_Datatype *staticProcessType);

int lastIteration(int chunk, int chunkSize, int input);

void sendPayload(int input, int chunkSize, double epsilon, MPI_Datatype staticProcessType, int target);

void workerProcess(int rank);

void masterProcess(int rank, int numOfProc, int input, double epsilon);