#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "matrix.h"

/**
 * @brief Error function to call when
 * something doesn't work.
 *
 * @param msg the message to print
 * @param arg any argument to print
 */
void error(char *msg, char *arg)
{
    fprintf(stderr, "%s %s\n", msg, arg);
    exit(EXIT_FAILURE);
}

/**
 * @brief Prints the matrix in the standard output
 *
 * @param m matrix structure
 */
void print_matrix(matrix *m)
{
    int i, j;
    char *coma = (char *)", ";

    printf("[");
    for (i = 0; i < m->rows; i++)
    {
        if (i > 0)
            printf(" ");
        printf("[");
        for (j = 0; j < m->columns; j++)
        {
            printf("%.6f%s", m->data[i * (m->columns) + j], j == m->columns - 1 ? "]" : coma);
        }
        if (i != m->rows - 1)
            printf(",\n");
    }

    printf("]\n");
}

/**
 * @brief Prints the first line of a matrix
 * (for debugging purposes)
 *
 * @param context a context
 * @param m the matrix
 */
void print_fl(char *context, matrix *m)
{
    printf("%s", context);
    int i;
    char *coma = (char *)", ";

    printf("[");
    for (i = 0; i < m->columns; i++)
    {
        printf("%.4f%s", m->data[i], i == m->columns - 1 ? "]" : coma);
    }
    printf("\n");
}

/**
 * @brief Prints the shape of a matrix
 * (for debugging purposes)
 *
 * @param context the context
 * @param m the matrix
 */
void print_shape(char *context, matrix *m)
{
    printf("%s: [%d, %d]\n", context, m->rows, m->columns);
}

/**
 * @brief Prints a loading bar
 *
 * @param context the context of the loading bar
 * @param progress
 */
void print_loading_bar(char *context, float progress)
{
    int val = (int)(progress * 100);

    int lpad = (int)(progress * BAR_WIDTH);

    int rpad = BAR_WIDTH - lpad;

    printf("\r%s %3d%% [%.*s%*s]", context, val, lpad, BAR_STRING, rpad, (char *)"");

    fflush(stdout);
}

/**
 * @brief Builds a matrix of the index of all maximums
 * in the source matrix
 *
 * Example:
 *  src = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
 *  dest = [2, 2, 2]
 *
 * @param y
 * @return matrix*
 */
void max_indexes(matrix *dest, matrix *y)
{
    int i, k;

    for (i = 0; i < y->rows; i++)
    {
        float max = y->data[i * y->columns];
        float index = 0;

        for (k = 1; k < y->columns; k++)
        {
            if (y->data[i * y->columns + k] > max)
            {
                max = y->data[i * y->columns + k];
                index = k;
            }
        }
        dest->data[i] = index;
    }
}

/**
 * @brief Computes the cross loss
 *
 * @param y the expected values
 * @param y_hat the predicted values
 * @return float
 */
float cross_loss(matrix *y, matrix *y_hat)
{
    int i;
    float loss = 0.0f;

    for (i = 0; i < y->rows * y->columns; i++)
    {
        loss += y_hat->data[i] * log(y->data[i]);
    }

    return -loss / y->rows;
}
