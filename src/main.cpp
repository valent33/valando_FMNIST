#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "utils.h"
#include "init.h"
#include "iostreams.h"
#include "matrix.h"
#include "activation.h"
#include "network.h"

#define EPOCHS 20
#define LAYER_1 200
#define LAYER_2 10
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f
#define ETA 0.001f
#define MINI_BATCH_SIZE 32

int main()
{
    time_t start, end;
    int i, j;
    float train_accuracy, test_accuracy;
    start = time(NULL);

    // Loading train and test data
    matrix *train_data = load_csv_vectors((char *)"data/fashion_mnist_train_vectors.csv", N_TRAIN_VECTORS);
    matrix *test_data = load_csv_vectors((char *)"data/fashion_mnist_test_vectors.csv", N_TEST_VECTORS);
    // Loading desired outputs
    matrix *train_labels = load_csv_labels((char *)"data/fashion_mnist_train_labels.csv", N_TRAIN_VECTORS);
    matrix *test_labels = load_csv_labels((char *)"data/fashion_mnist_test_labels.csv", N_TEST_VECTORS);
    // Minibatch matrices
    matrix *mini_batch = m_init(MINI_BATCH_SIZE, VECTOR_SIZE);
    matrix *mini_batch_t = m_init(VECTOR_SIZE, MINI_BATCH_SIZE);
    matrix *labels_batch = m_init(MINI_BATCH_SIZE, LABEL_SIZE);

    // Weights
    matrix *w1 = m_init(VECTOR_SIZE, LAYER_1);
    matrix *w1_c = m_init(VECTOR_SIZE, LAYER_1);
    matrix *w2 = m_init(LAYER_1, LAYER_2);
    matrix *w2_c = m_init(LAYER_1, LAYER_2);
    matrix *w2_t = m_init(LAYER_2, LAYER_1);

    // Biases
    matrix *b1 = m_init(1, LAYER_1);
    matrix *b1_c = m_init(1, LAYER_1);
    matrix *b2 = m_init(1, LAYER_2);
    matrix *b2_c = m_init(1, LAYER_2);
    matrix *b1_broad = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *b2_broad = m_init(MINI_BATCH_SIZE, LAYER_2);

    // Other matrices
    matrix *z1 = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *z1_temp = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *z1_prime = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *z2 = m_init(MINI_BATCH_SIZE, LAYER_2);
    matrix *z2_temp = m_init(MINI_BATCH_SIZE, LAYER_2);

    matrix *a1 = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *a1_t = m_init(LAYER_1, MINI_BATCH_SIZE);
    matrix *a2 = m_init(MINI_BATCH_SIZE, LAYER_2);

    matrix *d1 = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *d1_temp = m_init(MINI_BATCH_SIZE, LAYER_1);
    matrix *d2 = m_init(MINI_BATCH_SIZE, LAYER_2);

    matrix *dW1 = m_init(VECTOR_SIZE, LAYER_1);
    matrix *dW2 = m_init(LAYER_1, LAYER_2);
    matrix *dB1 = m_init(1, LAYER_1);
    matrix *dB2 = m_init(1, LAYER_2);
    matrix *dW1_temp = m_init(VECTOR_SIZE, LAYER_1);
    matrix *dW2_temp = m_init(LAYER_1, LAYER_2);
    matrix *dB1_temp = m_init(1, LAYER_1);
    matrix *dB2_temp = m_init(1, LAYER_2);
    matrix *m_dW1_temp = m_init(VECTOR_SIZE, LAYER_1);
    matrix *m_dW2_temp = m_init(LAYER_1, LAYER_2);
    matrix *m_dB1_temp = m_init(1, LAYER_1);
    matrix *m_dB2_temp = m_init(1, LAYER_2);
    matrix *m_dW1 = m_init(VECTOR_SIZE, LAYER_1);
    matrix *m_dW2 = m_init(LAYER_1, LAYER_2);
    matrix *m_dB1 = m_init(1, LAYER_1);
    matrix *m_dB2 = m_init(1, LAYER_2);
    matrix *dW2_p2 = m_init(LAYER_2, LAYER_1);
    matrix *dW1_p2 = m_init(VECTOR_SIZE, LAYER_1);
    matrix *dB2_p2 = m_init(1, LAYER_2);
    matrix *dB1_p2 = m_init(1, LAYER_1);
    matrix *v_dW2_temp = m_init(LAYER_2, LAYER_1);
    matrix *v_dW1_temp = m_init(VECTOR_SIZE, LAYER_1);
    matrix *v_dB2_temp = m_init(1, LAYER_2);
    matrix *v_dB1_temp = m_init(1, LAYER_1);
    matrix *v_dW2 = m_init(LAYER_2, LAYER_1);
    matrix *v_dW1 = m_init(VECTOR_SIZE, LAYER_1);
    matrix *v_dB2 = m_init(1, LAYER_2);
    matrix *v_dB1 = m_init(1, LAYER_1);
    matrix *v_dW2_sqrt = m_init(LAYER_1, LAYER_2);
    matrix *v_dW1_sqrt = m_init(VECTOR_SIZE, LAYER_1);
    matrix *v_dB2_sqrt = m_init(1, LAYER_2);
    matrix *v_dB1_sqrt = m_init(1, LAYER_1);
    matrix *v_dW2_inv = m_init(LAYER_1, LAYER_2);
    matrix *v_dW1_inv = m_init(VECTOR_SIZE, LAYER_1);
    matrix *v_dB2_inv = m_init(1, LAYER_2);
    matrix *v_dB1_inv = m_init(1, LAYER_1);
    matrix *m_dW2_final = m_init(LAYER_1, LAYER_2);
    matrix *m_dW1_final = m_init(VECTOR_SIZE, LAYER_1);
    matrix *m_dB2_final = m_init(1, LAYER_2);
    matrix *m_dB1_final = m_init(1, LAYER_1);

    init_network(w1, w2, b1, b2);

    for (j = 0; j < EPOCHS; j++)
    {
        printf("Epoch %d\n", j);
        for (i = 0; i < N_TRAIN_VECTORS; i += MINI_BATCH_SIZE)
        {
            // Taking the next mini-batch
            m_batch(mini_batch, train_data, i, MINI_BATCH_SIZE);
            m_batch(labels_batch, train_labels, i, MINI_BATCH_SIZE);

            // ========================== FORWARD PASS ==========================

            forward_pass(mini_batch, w1, b1, b1_broad, z1_temp, z1, a1, w2, b2, b2_broad, z2_temp, z2, a2, MINI_BATCH_SIZE);

            if (i == 0)
                print_fl((char *)"z2", z2);

            // ======================= BACKPROPAGATION =======================

            // D2 = (A2 - Y)
            m_sub(d2, a2, labels_batch);

            // D1 = (W2^T * D2) [hadamard product] leakyReLU'(Z1)
            m_transpose(w2_t, w2);
            m_mul(d1_temp, d2, w2_t);
            m_leakyReLU_prime(z1_prime, z1);
            m_hadamard(d1, z1_prime, d1_temp);

            // DW2 = D2 * A1^T
            m_transpose(a1_t, a1);
            m_mul(dW2, a1_t, d2);
            m_div_scalar(dW2, (float)MINI_BATCH_SIZE);

            // DW1 = D1 * X^T
            m_transpose(mini_batch_t, mini_batch);
            m_mul(dW1, mini_batch_t, d1);
            m_div_scalar(dW1, (float)MINI_BATCH_SIZE);

            // DB2 = broadcast^-1(D2)
            m_sum(dB2, d2, 0);
            m_div_scalar(dB2, (float)MINI_BATCH_SIZE);
            m_sum(dB1, d1, 0);
            m_div_scalar(dB1, (float)MINI_BATCH_SIZE);

            // Adam optimizer
            // Step 1: m_dW2 = beta1 * m_dW2 + (1 - beta1) * dW2
            m_copy(dW2_temp, dW2);
            m_mul_scalar(dW2_temp, (1.0f - BETA1));
            m_copy(dW1_temp, dW1);
            m_mul_scalar(dW1_temp, (1.0f - BETA1));
            m_copy(dB2_temp, dB2);
            m_mul_scalar(dB2_temp, (1.0f - BETA1));
            m_copy(dB1_temp, dB1);
            m_mul_scalar(dB1_temp, (1.0f - BETA1));
            m_copy(m_dW2_temp, m_dW2);
            m_copy(m_dW1_temp, m_dW1);
            m_copy(m_dB2_temp, m_dB2);
            m_copy(m_dB1_temp, m_dB1);
            m_mul_scalar(m_dW2_temp, BETA1);
            m_mul_scalar(m_dW1_temp, BETA1);
            m_mul_scalar(m_dB2_temp, BETA1);
            m_mul_scalar(m_dB1_temp, BETA1);
            m_add(m_dW2, m_dW2_temp, dW2_temp);
            m_add(m_dW1, m_dW1_temp, dW1_temp);
            m_add(m_dB2, m_dB2_temp, dB2_temp);
            m_add(m_dB1, m_dB1_temp, dB1_temp);

            // Step 2: v_dW2 = beta2 * v_dW2 + (1 - beta2) * dW2^2
            m_square(dW2_p2, dW2);
            m_square(dW1_p2, dW1);
            m_square(dB2_p2, dB2);
            m_square(dB1_p2, dB1);
            m_mul_scalar(dW2_p2, (1.0f - BETA2));
            m_mul_scalar(dW1_p2, (1.0f - BETA2));
            m_mul_scalar(dB2_p2, (1.0f - BETA2));
            m_mul_scalar(dB1_p2, (1.0f - BETA2));
            m_copy(v_dW2_temp, v_dW2);
            m_copy(v_dW1_temp, v_dW1);
            m_copy(v_dB2_temp, v_dB2);
            m_copy(v_dB1_temp, v_dB1);
            m_mul_scalar(v_dW2_temp, BETA2);
            m_mul_scalar(v_dW1_temp, BETA2);
            m_mul_scalar(v_dB2_temp, BETA2);
            m_mul_scalar(v_dB1_temp, BETA2);
            m_add(v_dW2, v_dW2_temp, dW2_p2);
            m_add(v_dW1, v_dW1_temp, dW1_p2);
            m_add(v_dB2, v_dB2_temp, dB2_p2);
            m_add(v_dB1, v_dB1_temp, dB1_p2);

            // Step 3: m_dW2 = m_dW2 / (1 - beta1^t) (same for v_mdW2)
            m_div_scalar(m_dW2, (1.0f - pow(BETA1, j + 1)));
            m_div_scalar(m_dW1, (1.0f - pow(BETA1, j + 1)));
            m_div_scalar(m_dB2, (1.0f - pow(BETA1, j + 1)));
            m_div_scalar(m_dB1, (1.0f - pow(BETA1, j + 1)));

            // Step 4: W2 = W2 - (lr / (sqrt(v_dW2) + eps)) * m_dW2
            m_sqrt(v_dW2_sqrt, v_dW2);
            m_sqrt(v_dW1_sqrt, v_dW1);
            m_sqrt(v_dB2_sqrt, v_dB2);
            m_sqrt(v_dB1_sqrt, v_dB1);
            m_add_scalar(v_dW2_sqrt, EPSILON);
            m_add_scalar(v_dW1_sqrt, EPSILON);
            m_add_scalar(v_dB2_sqrt, EPSILON);
            m_add_scalar(v_dB1_sqrt, EPSILON);
            m_inv(v_dW2_inv, v_dW2_sqrt);
            m_inv(v_dW1_inv, v_dW1_sqrt);
            m_inv(v_dB2_inv, v_dB2_sqrt);
            m_inv(v_dB1_inv, v_dB1_sqrt);
            m_mul_scalar(m_dW2, ETA);
            m_mul_scalar(m_dW1, ETA);
            m_mul_scalar(m_dB2, ETA);
            m_mul_scalar(m_dB1, ETA);
            m_hadamard(m_dW2_final, m_dW2, v_dW2_inv);
            m_hadamard(m_dW1_final, m_dW1, v_dW1_inv);
            m_hadamard(m_dB2_final, m_dB2, v_dB2_inv);
            m_hadamard(m_dB1_final, m_dB1, v_dB1_inv);

            m_copy(w2_c, w2);
            m_copy(w1_c, w1);
            m_copy(b2_c, b2);
            m_copy(b1_c, b1);

            // Update weights
            m_sub(w2, w2_c, m_dW2_final);
            m_sub(w1, w1_c, m_dW1_final);
            m_sub(b2, b2_c, m_dB2_final);
            m_sub(b1, b1_c, m_dB1_final);

            // Resetting matrices
            m_zeros(dW2_temp);
            m_zeros(dW1_temp);
            m_zeros(dB2_temp);
            m_zeros(dB1_temp);
            m_zeros(dW2_p2);
            m_zeros(dW1_p2);
            m_zeros(dB2_p2);
            m_zeros(dB1_p2);
            m_zeros(v_dW2_temp);
            m_zeros(v_dW1_temp);
            m_zeros(v_dB2_temp);
            m_zeros(v_dB1_temp);
            m_zeros(v_dW2_sqrt);
            m_zeros(v_dW1_sqrt);
            m_zeros(v_dB2_sqrt);
            m_zeros(v_dB1_sqrt);
            m_zeros(v_dW2_inv);
            m_zeros(v_dW1_inv);
            m_zeros(v_dB2_inv);
            m_zeros(v_dB1_inv);
            m_zeros(m_dW2_temp);
            m_zeros(m_dW1_temp);
            m_zeros(m_dB2_temp);
            m_zeros(m_dB1_temp);
            m_zeros(m_dW2_final);
            m_zeros(m_dW1_final);
            m_zeros(m_dB2_final);
            m_zeros(m_dB1_final);
            m_zeros(w2_c);
            m_zeros(w1_c);
            m_zeros(b2_c);
            m_zeros(b1_c);
            reset_network(dW1, dW2, dB1, dB2, z1, z2, a1, d1, d2, z1_temp, z2_temp);
        }
    }

    // ========================== TESTING ==========================
    train_accuracy = predict_output(train_data, train_labels, w1, w2, b1, b2, N_TRAIN_VECTORS, LAYER_1, LAYER_2, (char *)"trainPredictions");

    test_accuracy = predict_output(test_data, test_labels, w1, w2, b1, b2, N_TEST_VECTORS, LAYER_1, LAYER_2, (char *)"testPredictions");

    // ========================== FREE MEMORY ==========================

    m_free(train_data);
    m_free(test_data);
    m_free(test_labels);
    m_free(mini_batch);
    m_free(mini_batch_t);
    m_free(train_labels);
    m_free(labels_batch);
    m_free(w1);
    m_free(w1_c);
    m_free(w2);
    m_free(w2_c);
    m_free(w2_t);
    m_free(b1);
    m_free(b1_c);
    m_free(b2);
    m_free(b2_c);
    m_free(b1_broad);
    m_free(b2_broad);
    m_free(z1);
    m_free(z1_temp);
    m_free(z1_prime);
    m_free(z2);
    m_free(z2_temp);
    m_free(a1);
    m_free(a1_t);
    m_free(a2);
    m_free(d1);
    m_free(d1_temp);
    m_free(d2);
    m_free(dW1);
    m_free(dW2);
    m_free(dB1);
    m_free(dB2);
    m_free(m_dW1);
    m_free(m_dW2);
    m_free(m_dB1);
    m_free(m_dB2);
    m_free(m_dW1_final);
    m_free(m_dW2_final);
    m_free(m_dB1_final);
    m_free(m_dB2_final);
    m_free(v_dW1);
    m_free(v_dW2);
    m_free(v_dB1);
    m_free(v_dB2);
    m_free(v_dW1_sqrt);
    m_free(v_dW2_sqrt);
    m_free(v_dB1_sqrt);
    m_free(v_dB2_sqrt);
    m_free(v_dW1_inv);
    m_free(v_dW2_inv);
    m_free(v_dB1_inv);
    m_free(v_dB2_inv);
    m_free(v_dW1_temp);
    m_free(v_dW2_temp);
    m_free(v_dB1_temp);
    m_free(v_dB2_temp);
    m_free(m_dW1_temp);
    m_free(m_dW2_temp);
    m_free(m_dB1_temp);
    m_free(m_dB2_temp);
    m_free(dW2_temp);
    m_free(dW1_temp);
    m_free(dB2_temp);
    m_free(dB1_temp);
    m_free(dW2_p2);
    m_free(dW1_p2);
    m_free(dB2_p2);
    m_free(dB1_p2);

    printf((char *)"TRAINING ACCURACY = %.3f | TESTING ACCURACY = %.3f\n", train_accuracy * 100, test_accuracy * 100);

    end = time(NULL);

    printf("Time elapsed  %.3f minutes\n", (float)(end - start) / 60.0f);

    return 0;
}