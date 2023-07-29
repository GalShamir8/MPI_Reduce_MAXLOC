#include "sqrt.h"

int main(int argc, char **argv)
{
  int rank, numOfProc, input;
  double epsilon;

  init(&argc, argv, &numOfProc, &rank, &input, &epsilon);
  process(rank, numOfProc - 1, input, epsilon);
  MPI_Finalize();
  return 0;
}

void masterProcess(int rank, int numOfProc, int input, double epsilon)
{
  MPI_Datatype staticProcessType;
  create_type(&staticProcessType);
  int result[2] = {-3, -3};
  int localMax[2] = {-3, -3};
  int chunkSize = input / numOfProc;
  for (int chunk = chunkSize; chunk <= input; chunk += chunkSize)
  {
    // support left over in case that input not diveded by numOfProc
    if (lastIteration(chunk, chunkSize, input)) 
      sendPayload(chunk, chunkSize + (input % numOfProc), epsilon, staticProcessType, chunk / chunkSize);
    else
      sendPayload(chunk, chunkSize, epsilon, staticProcessType, chunk / chunkSize);
  }
  MPI_Reduce(localMax, result, 1, MPI_2INT, MPI_MAXLOC, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  printf("[static] number requiring max number of iterations: %d. (number of iterations: %d)\n",
    result[1],
    result[0]
  );
  MPI_Type_free(&staticProcessType);
}

void workerProcess(int rank)
{
  MPI_Datatype staticProcessType;
  create_type(&staticProcessType);
  StaticProcessType receivedBuffer;
  MPI_Recv(
      &receivedBuffer,
      1,
      staticProcessType,
      ROOT_PROCESS_RANK,
      ROOT_PROCESS_RANK,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  printf("rank: %d recived message input: %d, chunk size: %d, epsilon: %f\n",
        rank, receivedBuffer.input, receivedBuffer.chunkSize, receivedBuffer.epsilon);

  int localMax[2] = {-1, -1};
  int result[2];

  for (int n = (receivedBuffer.input - receivedBuffer.chunkSize + 1); n <= receivedBuffer.input; n++)
  {    
    int numOfIterations;
    heron(n, receivedBuffer.epsilon, &numOfIterations);    
    if (numOfIterations > localMax[0])
    {      
      localMax[0] = numOfIterations;
      localMax[1] = n;
    }
  }
  MPI_Reduce(localMax, result, 1, MPI_2INT, MPI_MAXLOC, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  MPI_Type_free(&staticProcessType);
}

void process(int rank, int numOfProc, int input, double epsilon)
{
  if (rank == ROOT_PROCESS_RANK)
    masterProcess(rank, numOfProc, input, epsilon);
  else
    workerProcess(rank);
}

void sendPayload(int input, int chunkSize, double epsilon, MPI_Datatype staticProcessType, int target)
{
  MPI_Send(
      &((StaticProcessType){input, chunkSize, epsilon}),
      1,
      staticProcessType,
      target,
      ROOT_PROCESS_RANK,
      MPI_COMM_WORLD);
  printf("Sent payload %d %f to rank: %d\n", input, epsilon, target);
}

int lastIteration(int chunk, int chunkSize, int input)
{
  return (input < chunk + chunkSize) || (input > chunk + chunkSize && input < chunk + (2 * chunkSize));
}

void create_type(MPI_Datatype *staticProcessType)
{
  int block_lengths[3] = {1, 1, 1};
  MPI_Aint displacements[3];
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

  StaticProcessType type;
  MPI_Aint base_address;
  MPI_Get_address(&type, &base_address);
  MPI_Get_address(&type.input, &displacements[0]);
  MPI_Get_address(&type.chunkSize, &displacements[1]);
  MPI_Get_address(&type.epsilon, &displacements[2]);

  for (int i = 0; i < 3; i++)
    displacements[i] = MPI_Aint_diff(displacements[i], base_address);

  MPI_Type_create_struct(3, block_lengths, displacements, types, staticProcessType);
  MPI_Type_commit(staticProcessType);
}