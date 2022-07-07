#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"

/**
 * @brief Squares each element of a matrix
 *
 * @param dest the destination matrix
 * @param src the source matrix
 */
void m_square(matrix *dest, matrix *src)
{
    int i;
    m_zeros(dest);
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = src->data[i] * src->data[i];
}

/**
 * @brief Applies the square root function
 * to each element of a matrix
 *
 * @param dest the destination matrix
 * @param src the source matrix
 */
void m_sqrt(matrix *dest, matrix *src)
{
    int i;
    m_zeros(dest);
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = sqrt(src->data[i]);
}

/**
 * @brief Computes inverse of each element of the
 * matrix
 *
 * @param dest the destination matrix
 * @param src the source matrix
 */
void m_inv(matrix *dest, matrix *src)
{
    int i;
    m_zeros(dest);
    for (i = 0; i < dest->rows * dest->columns; i++)
        dest->data[i] = 1.0f / src->data[i];
}

/**
 * @brief Applies leakyReLU to each element of the
 * matrix
 *
 * @param dest the destination matrix
 * @param src the source matrix
 */
void m_leakyReLU(matrix *dest, matrix *src)
{
    int i;
    m_zeros(dest);
    for (i = 0; i < src->rows * src->columns; i++)
        dest->data[i] = src->data[i] > 0 ? src->data[i] : 0.01f * src->data[i];
}

/**
 * @brief Applies the leakyReLu derivative to each
 * element of the matrix
 *
 * @param dest the destination matrix
 * @param src the source matrix
 */
void m_leakyReLU_prime(matrix *dest, matrix *src)
{
    int i;
    m_zeros(dest);
    for (i = 0; i < src->rows * src->columns; i++)
        dest->data[i] = src->data[i] > 0 ? 1.0f : 0.01f;
}

/**
 * @brief Applies the softmax activation function
 * to a matrix *
 *
 * @param m
 * @return matrix*
 */
void m_softmax(matrix *dest, matrix *src)
{
    int i, j;
    float max = 0.0f;
    float sum;

    for (i = 0; i < src->rows; i++)
    {
        // Finding maximum of the row
        for (j = 0; j < src->columns; j++)
        {
            if (src->data[i * src->columns + j] > max)
                max = src->data[i * src->columns + j];
        }

        // Subtracting the maximum from each element
        for (j = 0; j < src->columns; j++)
        {
            src->data[i * src->columns + j] -= max;
        }

        // Finding the sum of the row
        sum = 0.0f;
        for (j = 0; j < src->columns; j++)
        {
            sum += exp(src->data[i * src->columns + j]);
        }

        // Applying the softmax function
        for (j = 0; j < src->columns; j++)
        {
            dest->data[i * src->columns + j] = exp(src->data[i * src->columns + j]) / sum;
        }

        // Resetting the maximum
        max = 0.0f;
    }
}
