#ifndef MODEL_H
#define MODEL_H

#include "linalg.h"

// Constants
#define E_VAL 2.718f

float sigmoid(float x);
float sigmoid_d(float x);

typedef struct {
    Mat *w; // Weights matrix (Out_dim x In_dim)
    Mat *b; // Bias column vector (Out_dim x 1)
} Layer;

typedef struct {
    int n_layers;      // Number of weight layers (e.g., 3 for a 4-size topology)
    int max_epochs;    // Training iterations
    int batch_size;
    float lr;          // Learning rate (η)
    Layer *layers;     // Array of layers
} Model;

/**
 * Creates a new model based on a column matrix of layer sizes.
 * If sizes are [784, 128, 10], it creates 2 layers.
 */
Model *model_new(Mat *layers_dim, int max_epochs, float learning_rate);

/**
 * Standard training loop using Gradient Descent.
 * X: Input matrix (Features x Samples)
 * Y: Target matrix (Labels x Samples)
 */
void model_train(Model *model, Mat *X, Mat *Y, int batch_s);

void mat_softmax(Mat *a, Model *model, Mat *softmax);

/**
 * Predicts the output for a given input matrix.
 * Returns a NEW matrix that must be freed.
 */
Mat *model_predict(Model *model, Mat *X);

/**
 * Cleans up all memory associated with the model, including layers.
 */
void model_free(Model *model);

/**
 * Converts a standard label vector (1 x N) into 
 * a One-Hot encoded matrix (10 x N).
 */
Mat *one_hot_encode(Mat *labels, int num_classes);

/**
 * Calculates model classification accuracy for a given dataset
 */
double calculate_accuracy(Model *model, Mat *X, Mat *Y_one_hot);

#endif