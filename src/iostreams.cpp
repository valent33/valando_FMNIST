#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "matrix.h"
#include "iostreams.h"
#include "utils.h"

/**
 * @brief Loads a train or test vectors file of 784 values
 *
 * @param filename the filename
 * @param n_rows the number of rows in the CSV
 * @return matrix*
 */
matrix *load_csv_vectors(char *filename, int n_rows)
{
    FILE *csv_file;
    long file_size;
    char *buffer;

    // Token for parsing
    char *token;
    char *b_token;
    char *token_save;
    char *b_token_save;

    // Rows and columns
    int n_line = 0;
    int n_tok = 0;

    // Opening the file
    if ((csv_file = fopen(filename, "r")) == NULL)
        error((char *)"Error opening a CSV file", filename);

    // Getting the file size in bytes
    fseek(csv_file, 0, SEEK_END);
    file_size = ftell(csv_file);
    fseek(csv_file, 0, SEEK_SET);

    // Allocating memory for the line
    buffer = (char *)malloc(file_size + 1);
    if (buffer == NULL)
        error((char *)"Error allocating memory for the line", (char *)"");

    // Creating a list of matrixes
    matrix *t = m_init(n_rows, VECTOR_SIZE);

    // Reading CSV file
    if ((int)fread(buffer, sizeof(char), file_size, csv_file) != file_size)
        error((char *)"Error reading the CSV file", filename);

    // Closing the file
    fclose(csv_file);
    buffer[file_size] = '\0';

    // Putting each line of the buffer in our matrix using strtok_r()
    b_token = strtok_r(buffer, "\n", &b_token_save);
    while (b_token != NULL)
    {
        token = strtok_r(b_token, (char *)",", &token_save);
        while (token != NULL)
        {
            // Converting the string to a float
            t->data[n_line * (t->columns) + n_tok] = atof(token) / 255.0f;
            token = strtok_r(NULL, (char *)",", &token_save);
            n_tok++;
        }

        n_tok = 0;
        n_line++;

        b_token = strtok_r(NULL, "\n", &b_token_save);
    }

    // Freeing memory
    free(buffer);

    return t;
}

/**
 * @brief Loads a set of test or train labels
 * and converts them in matrix of 0s and a 1
 * at the index of the value of the label
 *
 * Example: label = 2
 * label_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
 *
 * @param filename the filename
 * @param n_rows the number of rows in the CSV
 * @return matrix*
 */
matrix *load_csv_labels(char *filename, int n_rows)
{
    FILE *csv_file;
    int n_line = 0;
    char *line;
    int val;
    int i;

    matrix *t = m_init(n_rows, LABEL_SIZE);
    if (t == NULL)
        error((char *)"Error while allocating memory for matrix structure", (char *)"");

    line = (char *)malloc(LAB_BUFF_SIZE + 1);
    if (line == NULL)
        error((char *)"Error allocating memory for line", (char *)"");

    // Opening the file
    if ((csv_file = fopen(filename, "r")) == NULL)
        error((char *)"Error opening a CSV file", filename);

    // Reading CSV file
    while (fgets(line, LAB_BUFF_SIZE, csv_file) != NULL)
    {
        line[LAB_BUFF_SIZE] = '\0';
        val = strtof(line, NULL);

        // Filling the matrix
        for (i = 0; i < LABEL_SIZE; i++)
        {
            if (i == val)
                t->data[n_line * LABEL_SIZE + i] = 1.0f;
            else
                t->data[n_line * LABEL_SIZE + i] = 0.0f;
        }
        n_line += 1;
    }

    return t;
}

/**
 * @brief Write the labels matrix in files
 *
 * @param predicted_labels
 * @param filename
 */
void log_results(matrix *predicted_labels, char *filename)
{
    FILE *file;
    int i;

    if ((file = fopen(filename, "w")) == NULL)
        error((char *)"Error opening a file", filename);

    for (i = 0; i < predicted_labels->rows; i++)
    {
        fprintf(file, "%d\n", (int)predicted_labels->data[i]);
    }

    fclose(file);
}