#ifndef NETWORK_H_
#define NETWORK_H_

void inner_potential(matrix *z, matrix *z_temp, matrix *x, matrix *w, matrix *b, matrix *b_broad, int training_size);

float get_accuracy(matrix *prediction, matrix *labels, matrix *one_hot);

float predict_output(matrix *test_data, matrix *test_labels, matrix *w1, matrix *w2, matrix *b1, matrix *b2, int batch_size, int layer1_size, int layer2_size, char *filename);

void forward_pass(matrix *input, matrix *w1, matrix *b1, matrix *b1_broad, matrix *z1_temp, matrix *z1, matrix *a1, matrix *w2, matrix *b2, matrix *b2_broad, matrix *z2_temp, matrix *z2, matrix *a2, int size);

void reset_network(matrix *dW1, matrix *dW2, matrix *dB1, matrix *dB2, matrix *z1, matrix *z2, matrix *a1, matrix *d1, matrix *d2, matrix *z1_temp, matrix *z2_temp);
#endif // NETWORK_H_