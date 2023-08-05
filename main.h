#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

extern const int ROOT_PROCESS_RANK;

int main(int argc, char **argv);
void init(int *argc, char **argv, int *numOfProc, int *rank, int *input, double *epsilon);
void initSerial(int *argc, char **argv, int *numOfProc, int *rank, int *input, double *epsilon);
void dinit(int *argc, char **argv, int *numOfProc, int *rank, int *input, double *epsilon, int *chunkSize);
void mpiInit(int *argc, char **argv, int *numOfProc, int *rank);
void setInputs(int *input, double *eplsilon, int *argc, char **argv);
void dsetInputs(int *input, double *epsilon, int *chunkSize, int *argc, char **argv);
double heron(int n, double epsilon, int *counter);