#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "utils.h"

/**
 * @brief Initializes the weights of the neural network
 * accodring to the He initialization method
 *
 * @param m the matrix to initialize
 * @return matrix*
 */
void init_weights(matrix *m)
{
    int i;
    float r1, r2, normal_rand;

    for (i = 0; i < m->rows * m->columns; i++)
    {
        // Box Muller transofmration to get a random from a normal distribution
        r1 = (float)rand() / (float)RAND_MAX;
        r2 = (float)rand() / (float)RAND_MAX;
        normal_rand = sqrt(-2.0f * log(r1)) * cos(2.0f * M_PI * r2);

        m->data[i] = normal_rand * sqrt(2 / (float)m->rows);
    }
}

/**
 * @brief Initializes the biases
 *
 * @param m the matrix to initialize
 * @return matrix*
 */
void init_bias(matrix *m)
{
    int i, j;

    for (i = 0; i < m->rows; i++)
    {
        for (j = 0; j < m->columns; j++)
        {
            // Biases between 0 and 1
            m->data[i * (m->columns) + j] = 0.0f;
        }
    }
}

/**
 * @brief Initializes the weights and biases of the neural network
 *
 * @param w1 weights of the layer 1
 * @param w2 weights of the layer 2
 * @param b1 the biases of the layer 1
 * @param b2 the biases of the layer 2
 */
void init_network(matrix *w1, matrix *w2, matrix *b1, matrix *b2)
{
    init_weights(w1);
    init_weights(w2);
    init_bias(b1);
    init_bias(b2);
}
