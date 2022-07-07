#ifndef INIT_H_
#define INIT_H_
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

void init_weights(matrix *m);

void init_bias(matrix *m);

void init_network(matrix *w1, matrix *w2, matrix *b1, matrix *b2);

#endif // INIT_H_