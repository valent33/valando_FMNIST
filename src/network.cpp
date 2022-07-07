#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "network.h"
#include "iostreams.h"
#include "matrix.h"
#include "network.h"
#include "utils.h"
#include "activation.h"

/**
 * @brief Calculates the inner potential
 * of a matrix of neurons
 *
 * @param output the output where the inner potential result is stored
 * @param input the inputs
 * @param weights the weights
 * @param bias the biases
 * @param bias_broad a matrix where the biases are goign to be broadcasted
 * @param training_size the batch training size (for the biases to be broadcasted)
 */
void inner_potential(matrix *z, matrix *z_temp, matrix *x, matrix *w, matrix *b, matrix *b_broad, int training_size)
{
    m_mul(z_temp, x, w);
    m_broadcast(b_broad, b, training_size);
    m_add(z, z_temp, b_broad);
}

/**
 * @brief Computes the forward pass of the network
 *
 * @param input the inputs
 * @param w1 the weights of layer 1
 * @param b1 the weights of layer 2
 * @param b1_broad the biases of layer 1 broadcast matrix (to be filled)
 * @param z1 the inner potential matrix
 * @param a1 the activation matrix
 * @param w2 the weights of the layer 2
 * @param b2 the baises of the layer 2
 * @param b2_broad the biases of the layer 2 broadcast matrix (to be filled)
 * @param z2 the inner potential matrix
 * @param a2 the activation matrix
 * @param size the size of the batch
 */
void forward_pass(matrix *input, matrix *w1, matrix *b1, matrix *b1_broad, matrix *z1_temp, matrix *z1, matrix *a1, matrix *w2, matrix *b2, matrix *b2_broad, matrix *z2_temp, matrix *z2, matrix *a2, int size)
{
    // First layer forward pass
    inner_potential(z1, z1_temp, input, w1, b1, b1_broad, size);
    m_leakyReLU(a1, z1);

    // Second layer forward pass
    inner_potential(z2, z2_temp, a1, w2, b2, b2_broad, size);
    m_softmax(a2, z2);
}

void reset_network(matrix *dW1, matrix *dW2, matrix *dB1, matrix *dB2, matrix *z1, matrix *z2, matrix *a1, matrix *d1, matrix *d2, matrix *z1_temp, matrix *z2_temp)
{
    m_zeros(dW1);
    m_zeros(dW2);
    m_zeros(dB1);
    m_zeros(dB2);
    m_zeros(z1);
    m_zeros(z2);
    m_zeros(a1);
    m_zeros(d1);
    m_zeros(d2);
    m_zeros(z1_temp);
    m_zeros(z2_temp);
}

/**
 * @brief Get the accuracy of a prediction matrix
 * based on the labels matrix
 *
 * @param prediction the matrix containing predictions (one hot)
 * @param labels the matrix containing the labels (one hot)
 * @return float (the accuracy)
 */
float get_accuracy(matrix *prediction, matrix *labels, matrix *one_hot)
{
    float acc = 0;
    int i;
    m_one_hot(one_hot, prediction);

    for (i = 0; i < prediction->rows; i++)
    {
        if (m_row_cmp(one_hot, labels, i) == 0)
            acc++;
    }

    return acc / (float)prediction->rows;
}
/**
 * @brief Predicts the output of a test set
 *
 * @param test_data the matrix of training data
 * @param test_labels the matrix of the labels of this data
 * @param w1 the weights of the first layer
 * @param w2 the weights of the second layer
 * @param b1 the biases of the first layer
 * @param b2 the biases of the second layer
 * @param batch_size the batch size
 * @param layer1_size the size of the first layer
 * @param layer2_size the size of the second layer
 * @param filename the filename where to store the predictions
 * @return float
 */
float predict_output(matrix *test_data, matrix *test_labels, matrix *w1, matrix *w2, matrix *b1, matrix *b2, int batch_size, int layer1_size, int layer2_size, char *filename)
{
    float accuracy = 0;
    matrix *z1 = m_init(batch_size, layer1_size);
    matrix *z1_temp = m_init(batch_size, layer1_size);
    matrix *z2 = m_init(batch_size, layer2_size);
    matrix *z2_temp = m_init(batch_size, layer2_size);
    matrix *a1 = m_init(batch_size, layer1_size);
    matrix *a2 = m_init(batch_size, layer2_size);
    matrix *b1_broad = m_init(batch_size, layer1_size);
    matrix *b2_broad = m_init(batch_size, layer2_size);
    matrix *a2_oh = m_init(batch_size, layer2_size);
    matrix *a2_predict = m_init(batch_size, 1);

    forward_pass(test_data, w1, b1, b1_broad, z1_temp, z1, a1, w2, b2, b2_broad, z2_temp, z2, a2, batch_size);

    // Transforming the output to one hot
    accuracy = get_accuracy(a2, test_labels, a2_oh);

    // Writing test in file
    max_indexes(a2_predict, a2_oh);
    log_results(a2_predict, filename);

    m_free(z1);
    m_free(z2);
    m_free(a1);
    m_free(a2);
    m_free(b1_broad);
    m_free(b2_broad);
    m_free(a2_oh);
    m_free(a2_predict);
    m_free(z2_temp);
    m_free(z1_temp);

    return accuracy;
}
