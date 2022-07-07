#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_
#include "matrix.h"

void m_square(matrix *dest, matrix *src);

void m_sqrt(matrix *dest, matrix *src);

void m_inv(matrix *dest, matrix *src);

void m_leakyReLU(matrix *dest, matrix *src);

void m_leakyReLU_prime(matrix *dest, matrix *src);

void m_softmax(matrix *dest, matrix *src);

#endif // ACTIVATIONS_H_