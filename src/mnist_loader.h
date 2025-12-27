#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include "linalg.h"

Mat *read_mnist_images(const char *filename);
Mat *read_mnist_labels(const char *filename);

#endif
