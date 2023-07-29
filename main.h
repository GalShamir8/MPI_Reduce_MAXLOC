#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

extern const int ROOT_PROCESS_RANK;
extern const int MIN_NUM_OF_PROC;
extern const int DEFAULT_INPUT;
extern const double DEFAULT_EPSILON;

int main(int argc, char **argv);
void init(int *argc, char **argv, int *numOfProc, int *rank, int *input, double *epsilon);
void mpiInit(int *argc, char **argv, int *numOfProc, int *rank);
void setInputs(int *input, double *eplsilon, int *argc, char **argv);
double heron(int n, double epsilon, int *counter);