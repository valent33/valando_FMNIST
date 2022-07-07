#ifndef MATRIX_H_
#define MATRIX_H_
#include <stdbool.h>

typedef struct matrix
{
    int rows;
    int columns;
    float *data;
} matrix;

matrix *m_init(int nb_rows, int nb_columns);

void m_zeros(matrix *dest);

void m_transpose(matrix *dest, matrix *src);

void m_add(matrix *dest, matrix *src1, matrix *src2);

void m_sub(matrix *dest, matrix *m1, matrix *m2);

void m_mul_scalar(matrix *dest, float s);

void m_div_scalar(matrix *dest, float scalar);

void m_add_scalar(matrix *dest, float scalar);

void m_mul(matrix *dest, matrix *m1, matrix *m2);

void m_hadamard(matrix *dest, matrix *m1, matrix *m2);

void m_sum(matrix *dest, matrix *m, int axis);

void m_one_hot(matrix *dest, matrix *src);

int m_row_cmp(matrix *m1, matrix *m2, int row);

void m_broadcast(matrix *dest, matrix *bias, int rows);

void m_free(matrix *m);

void m_copy(matrix *dest, matrix *src);

void m_batch(matrix *dest, matrix *src, int start, int n_rows);

#endif // MATRIX_H_