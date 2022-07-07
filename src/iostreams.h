#ifndef IOSTREAMS_H_
#define IOSTREAMS_H_
#include "matrix.h"
#include <stddef.h>

// Length of the test/train vectors = 28x28
// Length of the labels vector = 10
// They are built as a list of 0s with a 1 in the position of the label value
#define VECTOR_SIZE 784
#define LABEL_SIZE 10

// Buffer sizes for reading CSV files
#define VEC_BUFF_SIZE 4704
#define LAB_BUFF_SIZE 5

// Number of test and train vectors
#define N_TEST_VECTORS 10000
#define N_TRAIN_VECTORS 60000

matrix *load_csv_vectors(char *filename, int n_rows);

matrix *load_csv_labels(char *filename, int n_rows);

void log_results(matrix *predicted_labels, char *filename);

#endif // IOSTREAMS_H_