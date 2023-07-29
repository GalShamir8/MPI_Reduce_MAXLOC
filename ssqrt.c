# include "ssqrt.h"

int main(int argc, char **argv)
{
  int rank, numOfProc, input;
  double epsilon;

  init(&argc, argv, &numOfProc, &rank, &input, &epsilon);
  // printf("rank: %d numOfProc: %d input: %d epsilon: %f\n", rank, numOfProc, input, epsilon);
  process(rank, numOfProc - 1, input, epsilon);
  MPI_Finalize();
  return 0;
}

void process(int rank, int numOfProc, int input, double epsilon)
{
  MPI_Datatype staticProcessType;
  create_type(&staticProcessType);
  Result sendbuf;
  Result recvbuf;
  if (rank == ROOT_PROCESS_RANK)
  {
    printf("In root rank process\n");
    int chunkSize = input / numOfProc;

    for (int chunk = chunkSize; chunk <= input; chunk += chunkSize) 
    {
      // support left over in case that input not diveded by numOfProc
      if (lastIteration(chunk, chunkSize, input)) { chunkSize += (input % numOfProc); };

      sendPayload(chunk, chunkSize, epsilon, staticProcessType, chunk / chunkSize);
    }
    MPI_Reduce(&sendbuf, &recvbuf, 1, MPI_DOUBLE_INT, MPI_MAXLOC, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
    printf(
      "number requiring max number of iterations: %f. (number of iterations: %d)\n", 
      recvbuf.max, 
      recvbuf.iterations
    );    
  } else
  {
    printf("In rank: %d process\n", rank);
    StaticProcessType receivedBuffer;    
    double heronResult;    

    MPI_Recv(
      &receivedBuffer, 
      1, 
      staticProcessType, 
      ROOT_PROCESS_RANK, 
      ROOT_PROCESS_RANK, 
      MPI_COMM_WORLD, 
      MPI_STATUS_IGNORE
    );
    printf(
      "rank: %d recived message input: %d, chunk size: %d, epsilon: %f\n", 
      rank, receivedBuffer.input, receivedBuffer.chunkSize, receivedBuffer.epsilon
    );    
    for (int n = receivedBuffer.input; n > (receivedBuffer.input - receivedBuffer.chunkSize + 1); n--)  
    {
      heronResult = heron(n, receivedBuffer.epsilon, &sendbuf.iterations);
      printf("heron(%d) result: %f, %d\n",n, heronResult, sendbuf.iterations);
      sendbuf.max = heronResult;
      MPI_Reduce(&sendbuf, &recvbuf, 1, MPI_DOUBLE_INT, MPI_MAXLOC, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
    }
  }
  MPI_Type_free(&staticProcessType);  
}

void sendPayload(int input, int chunkSize, double epsilon, MPI_Datatype staticProcessType, int target)
{
  MPI_Send(
    &((StaticProcessType) {input, chunkSize, epsilon}),
    1,
    staticProcessType, 
    target, 
    ROOT_PROCESS_RANK, 
    MPI_COMM_WORLD
  );
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