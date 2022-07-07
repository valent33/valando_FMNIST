#ifndef UTILS_H_
#define UTILS_H_
#include "matrix.h"

#define BAR_STRING "################################################################################"
#define BAR_WIDTH 80

void error(char *msg, char *arg);

void print_matrix(matrix *m);

void print_shape(char *arg, matrix *m);

void print_fl(char *context, matrix *m);

void print_loading_bar(char *context, float progress);

matrix *normalize_data(matrix *m);

void max_all_index(matrix *dest, matrix *y);

void max_indexes(matrix *dest, matrix *y);

#endif // UTILS_H_c