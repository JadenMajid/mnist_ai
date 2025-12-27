#include "mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Helper to handle MNIST Big-Endian format
static int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat *read_mnist_images(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    int magic, num_images, num_rows, num_cols;
    fread(&magic, sizeof(int), 1, fp);
    fread(&num_images, sizeof(int), 1, fp);
    fread(&num_rows, sizeof(int), 1, fp);
    fread(&num_cols, sizeof(int), 1, fp);

    num_images = reverse_int(num_images);
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);

    int img_size = num_rows * num_cols;
    // We store images as rows: (60000 x 784)
    Mat *images = mat_new(num_images, img_size); 

    unsigned char *buf = malloc(img_size);
    for (int i = 0; i < num_images; i++) {
        fread(buf, sizeof(unsigned char), img_size, fp);
        for (int j = 0; j < img_size; j++) {
            // Indexing: i (row) * img_size (width) + j (column)
            images->data[i * img_size + j] = (float)buf[j] / 255.0f;
        }
    }

    free(buf);
    fclose(fp);
    return images;
}

Mat *read_mnist_labels(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    int magic, num_labels;
    fread(&magic, sizeof(int), 1, fp);
    fread(&num_labels, sizeof(int), 1, fp);
    num_labels = reverse_int(num_labels);

    // Create One-Hot Encoded Matrix: (10 rows x 60,000 columns)
    Mat *labels = mat_new(num_labels, 10);
    mat_fill(labels, 0.0f);

    unsigned char label;
    for (int i = 0; i < num_labels; i++) {
        fread(&label, sizeof(unsigned char), 1, fp);
        if (label < 10) {
            // Indexing: i (row) * 10 (width) + label (column)
            labels->data[i * 10 + (int)label] = 1.0f; 
        }
    }

    fclose(fp);
    return labels;
}