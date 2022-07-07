#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>
#include "matrix.h"
#include "utils.h"
void m_zeros(matrix *dest)
{
    int i;
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = 0.0f;
}
/**
 * @brief Initializes an empty matrix structure with
 * a specified number of rows and columns
 *
 * @param nb_rows int : number of rows
 * @param nb_columns int : number of columns
 * @return matrix* matrix structure
 */
matrix *m_init(int nb_rows, int nb_columns)
{
    if (nb_rows <= 0 || nb_columns <= 0)
    {
        error((char *)"Invalid matrix size when initializing matrix", (char *)"");
    }

    // Allocating structure
    matrix *m = (matrix *)malloc(sizeof(matrix));
    if (m == NULL)
        error((char *)"Error while allocating memory for matrix structure", (char *)"");

    m->rows = nb_rows;
    m->columns = nb_columns;

    // Allocating for data
    m->data = (float *)calloc(sizeof(float), nb_rows * nb_columns);
    if (m->data == NULL)
        error((char *)"Error while allocating memory for matrix data", (char *)"");

    return m;
}

/**
 * @brief Fills a matrix with 0s
 *
 * @param dest the destination matrix
 */
void m_zeros(matrix *dest)
{
    int i;
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = 0.0f;
}

/**
 * @brief Transposes a matrix
 *
 * @param dest where the result goes
 * @param m the matrix to transpose
 */
void m_transpose(matrix *dest, matrix *src)
{
    if (dest->rows != src->columns || dest->columns != src->rows)
        error((char *)"m_transpose(): Error when transposing matrices", (char *)"");

    int i, j;
#pragma omp parallel for private(i, j) shared(dest, src)
    for (i = 0; i < src->rows; i++)
        for (j = 0; j < src->columns; j++)
            dest->data[j * src->rows + i] = src->data[i * src->columns + j];
}

/**
 * @brief Adds two matrices and put the result
 * in the first one
 *
 * @param dest the first matrix, where the result goes
 * @param src the second matrix
 */
void m_add(matrix *dest, matrix *src1, matrix *src2)
{
    if (dest->rows != src1->rows || dest->columns != src1->columns || src1->rows != src2->rows || src1->columns != src2->columns)
        error((char *)"m_add(): Error when adding matrices", (char *)"");

    int i;
    m_zeros(dest);
#pragma omp parallel for private(i) shared(dest, src1, src2)
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = src1->data[i] + src2->data[i];
}

/**
 * @brief Substracts two matrices and put the result
 * in a third one
 *
 * @param dest the matrix where the result goes
 * @param m1 the first matrix
 * @param m2 the second matrix
 */
void m_sub(matrix *dest, matrix *m1, matrix *m2)
{
    if (m1->rows != m2->rows || m2->columns != m1->columns)
    {
        print_shape((char *)"m1", m1);
        print_shape((char *)"m2", m2);
        error((char *)"m_sub(): Error when substracting matrices", (char *)"");
    }

    int i;
    m_zeros(dest);
#pragma omp parallel for private(i) shared(dest, m1, m2)
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = m1->data[i] - m2->data[i];
}

/**
 * @brief Multiplies a matrix by a scalar
 *
 * @param dest the matrix to mulitply, where the result goes
 * @param s the scalar (float)
 */
void m_mul_scalar(matrix *dest, float s)
{
    int i;
#pragma omp parallel for private(i) shared(dest)
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] *= s;
}

/**
 * @brief Divides a matrix by a scalar
 *
 * @param dest the destination matrix
 * @param scalar the scalar
 */
void m_div_scalar(matrix *dest, float scalar)
{
    int i;
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = dest->data[i] / (scalar + 1e-4);
}

/**
 * @brief Adds a scalar to a matrix
 *
 * @param dest the destination matrix
 * @param scalar the scalar
 */
void m_add_scalar(matrix *dest, float scalar)
{
    int i;
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] += scalar;
}

/**
 * @brief Multiplies a matrix by another and store the
 * result in a third one
 *
 * @param dest the matrix where the result goes
 * @param m1 the first matrix
 * @param m2 the second matrix
 */
void m_mul(matrix *dest, matrix *m1, matrix *m2)
{
    if (m1->columns != m2->rows || dest->rows != m1->rows || dest->columns != m2->columns)
        error((char *)"m_mul(): Error when multiplying matrices", (char *)"");

    int i, j, k;
    m_zeros(dest);
#pragma omp parallel for private(i, j, k) shared(dest, m1, m2)
    for (i = 0; i < dest->rows; i++)
        for (k = 0; k < m1->columns; k++)
        {
            for (j = 0; j < dest->columns; j++)
                dest->data[i * dest->columns + j] += m1->data[i * m1->columns + k] * m2->data[k * m2->columns + j];
        }
}

/**
 * @brief Hadamard matrix multiplication, multiplies
 * a matrix by another and store the result in a third one
 *
 * @param dest the matrix where the result goes
 * @param m1 the first matrix
 * @param m2 the second matrix
 */
void m_hadamard(matrix *dest, matrix *m1, matrix *m2)
{
    if (dest->rows != m1->rows || dest->columns != m1->columns || dest->rows != m2->rows || dest->columns != m2->columns)
        error((char *)"m_hadamard(): Error when multiplying matrices", (char *)"");

    m_zeros(dest);
    int i;
#pragma omp parallel for private(i) shared(dest, m1, m2)
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = m1->data[i] * m2->data[i];
}

/**
 * @brief Sums a matrix along an axis. Similar to
 * numpy.sum()
 *
 * @param dest the destination matrix
 * @param m the source matrix
 * @param axis the axis along which to sum
 */
void m_sum(matrix *dest, matrix *m, int axis)
{
    int i, j;
    m_zeros(dest);

    // If axis = 0 sum along rows
    if (axis == 0)
    {
        if (dest->rows != 1 || dest->columns != m->columns)
            error((char *)"m_sum(): Error when suming along rows", (char *)"");

#pragma omp parallel for private(i, j) shared(dest, m)
        for (j = 0; j < m->columns; j++)
        {
            for (i = 0; i < m->rows; i++)
                dest->data[j] += m->data[i * m->columns + j];
        }
    }
    else
    {
        if (dest->rows != m->rows || dest->columns != 1)
            error((char *)"m_sum(): Error when suming along columns", (char *)"");

#pragma omp parallel for private(i, j) shared(dest, m)
        for (i = 0; i < m->rows; i++)
        {
            for (j = 0; j < m->columns; j++)
                dest->data[i] += m->data[i * m->columns + j];
        }
    }
}

/**
 * @brief Computes the maximum index in each row of
 * the matrix, and then transforms the matrix with
 * 0 and 1, where 1 is the maximum value in the row
 *
 * @param src the source matrix
 * @param dest the destination matrix
 * @return matrix*
 */
void m_one_hot(matrix *dest, matrix *src)
{
    float max;
    int max_index;
    int i, j;
    m_zeros(dest);

    // Find the maximum value in each row
#pragma omp parallel for private(i, j, max, max_index) shared(dest, src)
    for (i = 0; i < src->rows; i++)
    {
        max = src->data[i * src->columns];
        max_index = 0;
        for (j = 1; j < src->columns; j++)
        {
            if (src->data[i * src->columns + j] > max)
            {
                max = src->data[i * src->columns + j];
                max_index = j;
            }
        }

        // Set the maximum value to 1
        dest->data[i * dest->columns + max_index] = 1.0f;
    }
}

/**
 * @brief Compares two rows and returns 0
 * if they are equal, 1 otherwise
 *
 * @param m1 the first matrix
 * @param m2 the second matrix
 * @param r the row to compare
 * @return int
 */
int m_row_cmp(matrix *m1, matrix *m2, int row)
{
    int i;

    for (i = 0; i < m1->columns; i++)
    {
        if (m1->data[row * (m1->columns) + i] != m2->data[row * (m2->columns) + i])
            return 1;
    }
    return 0;
}

/**
 * @brief Broadcasts the bias matrix (1, x) matrix to a
 * (n, x) matrix by copying all the lines
 *
 * @param bias The bias vector
 * @param rows The number of rows of the larger matrix
 * @return matrix* result
 */
void m_broadcast(matrix *dest, matrix *bias, int rows)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < dest->columns; j++)
        {
            dest->data[i * dest->columns + j] = bias->data[j];
        }
    }
}

/**
 * @brief Frees a matrix
 *
 * @param m the matrix
 */
void m_free(matrix *m)
{
    if (m == NULL)
        error((char *)"Error while freeing matrix: matrix is already NULL", (char *)"");

    free(m->data);
    free(m);
}

/**
 * @brief Copies a source matrix in a destination
 * matrix
 *
 * @param dest the destination matrix
 * @param src the source matrix
 */
void m_copy(matrix *dest, matrix *src)
{
    if (dest->rows != src->rows || dest->columns != src->columns)
        error((char *)"Error while copying matrix: wrong dimensions", (char *)"");

    int i;
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = src->data[i];
}

/**
 * @brief Takes a number of rows of the matrix and
 * puts them in another
 * @param dest the destination matrix
 * @param src the source matrix
 * @param start the row where to start
 * @param end the number of rows to take
 */
void m_batch(matrix *dest, matrix *src, int start, int n_rows)
{
    int i, j;
    // Copying n_rows rows of the source matrix starting at start
    for (i = 0; i < n_rows; i++)
    {
        for (j = 0; j < src->columns; j++)
        {
            dest->data[i * src->columns + j] = src->data[(start + i) * src->columns + j];
        }
    }
}
