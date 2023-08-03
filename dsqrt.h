# include "main.h"

enum tags {WORK, STOP};

typedef struct
{
  int input;
  int chunkSize;    
  double epsilon;
} StaticProcessType;

void process(int rank, int numOfProc, int input, double epsilon, int chunkSize);

void workerProcess(int rank);

void masterProcess(int rank, int numOfProc, int input, double epsilon, int chunkSize);

void create_type(MPI_Datatype *staticProcessType);

void sendPayload(int input, int chunkSize, double epsilon, MPI_Datatype staticProcessType, int target, int tag);