#include "dsqrt.h"

int main(int argc, char **argv)
{
  int rank, numOfProc, input, chunkSize;
  double epsilon;

  dinit(&argc, argv, &numOfProc, &rank, &input, &epsilon, &chunkSize);	
  process(rank, numOfProc - 1, input, epsilon, chunkSize);
  MPI_Finalize();
  return 0;
}

void masterProcess(int rank, int numOfProc, int input, double epsilon, int chunkSize)
{
  MPI_Datatype staticProcessType;
  MPI_Status status;
  create_type(&staticProcessType);
  int result[2] = {-3, -3};
  int localMax[2] = {-3, -3};
  int tasksDoneCount = 0, chunk = chunkSize, numOfTasks = (input / chunkSize), taskSent = 0, payload, payloadSize;

  while(tasksDoneCount < numOfTasks)
  {
    if (taskSent == numOfTasks){
      int recevied;
      MPI_Recv(&recevied, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      tasksDoneCount++;
      sendPayload(0, 0, 0.0, staticProcessType, status.MPI_SOURCE, STOP);
    } else
    {
      if ((numOfTasks - tasksDoneCount) == 1)
      {
        payload = input;
        payloadSize = input - chunk;
      }
      else
      {
        payload = chunk;
        payloadSize = chunkSize;
      }
      
      if ((chunk / chunkSize) > numOfProc)
      {
        int recevied;
        MPI_Recv(&recevied, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        tasksDoneCount++;
        chunk += chunkSize;
        sendPayload(payload, payloadSize, epsilon, staticProcessType, status.MPI_SOURCE, WORK);
        taskSent++;
      } else {
        sendPayload(payload, payloadSize, epsilon, staticProcessType, chunk/chunkSize, WORK);
        taskSent++;
        chunk += chunkSize;
      }
    }
  }

  MPI_Reduce(localMax, result, 1, MPI_2INT, MPI_MAXLOC, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  printf("[dynamic] number requiring max number of iterations: %d. (number of iterations: %d)\n",
         result[1],
         result[0]);
  MPI_Type_free(&staticProcessType);
}

void workerProcess(int rank)
{
  StaticProcessType receivedBuffer;
  MPI_Status status;
  MPI_Datatype staticProcessType;
  create_type(&staticProcessType);
  int tag;
  int localMax[2] = {-1, -1};
  int result[2];  
  do
  {
    tag = STOP;
    MPI_Recv(
      &receivedBuffer,
      1,
      staticProcessType,
      ROOT_PROCESS_RANK,
      MPI_ANY_TAG,
      MPI_COMM_WORLD,
      &status
    );

    tag = status.MPI_TAG;
    if (tag == WORK)
    {
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
      MPI_Send(0, 0, MPI_INT, ROOT_PROCESS_RANK, STOP, MPI_COMM_WORLD);
    }
  } while (tag != STOP);

  MPI_Reduce(localMax, result, 1, MPI_2INT, MPI_MAXLOC, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  MPI_Type_free(&staticProcessType);
}

void process(int rank, int numOfProc, int input, double epsilon, int chunkSize)
{
  if (rank == ROOT_PROCESS_RANK)
  {
    double start = MPI_Wtime();
    masterProcess(rank, numOfProc, input, epsilon, chunkSize);
    printf("Runtime: %lf\n", MPI_Wtime() - start);
  }
  else
    workerProcess(rank);
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

void sendPayload(int input, int chunkSize, double epsilon, MPI_Datatype staticProcessType, int target, int tag)
{
  MPI_Send(
    &((StaticProcessType){input, chunkSize, epsilon}),
    1,
    staticProcessType,
    target,
    tag,
    MPI_COMM_WORLD
  );
}