#include "model.h"
#include <math.h>

#define PRINT_FREQUENCY 500

// --- Activation Functions ---
float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

float sigmoid_d(float x)
{
  float s = sigmoid(x);
  return s * (1.0f - s);
}

float relu(float x)
{
  return (x < 0.0) ? 0.0 : x;
}

float relu_d(float x)
{
  return (x < 0.0) ? 0.0 : 1.0;
}

Model* model_new(Mat* layers_dim, int max_epochs, float learning_rate)
{
  Model* model = (Model*)malloc(sizeof(Model));
  model->max_epochs = max_epochs;
  model->lr = learning_rate;
  model->n_layers = (int)layers_dim->rows; // Correcting to use rows from topology
  model->layers = malloc(sizeof(Layer) * (model->n_layers - 1));

  for (int i = 0; i < model->n_layers - 1; i++)
  {
    float n_in = layers_dim->data[i];
    float n_out = layers_dim->data[i + 1];
    float randomization_max = sqrt(2 / (n_in));

    // Bias = (1 x n_out) Row Vector
    model->layers[i].b = mat_new(1, (int)n_out);
    mat_rand(model->layers[i].b, -randomization_max, randomization_max);

    // Weights = (n_in x n_out)
    model->layers[i].w = mat_new((int)n_in, (int)n_out);
    mat_rand(model->layers[i].w, -randomization_max, randomization_max);
  }

  return model;
}
void model_train(Model* model, Mat* X, Mat* Y_labels, int batch_s)
{
  int n = X->rows;        // 60,000 samples
  int d = X->cols;        // 784 features
  int c = Y_labels->cols; // 10 classes

  // --- 1. HOISTED ALLOCATIONS ---
  Mat* a = malloc(model->n_layers * sizeof(Mat));
  Mat* a_T = malloc(model->n_layers * sizeof(Mat));
  Mat* z = malloc(model->n_layers * sizeof(Mat));
  Mat* w_T = malloc((model->n_layers - 1) * sizeof(Mat)); // Only n-1 weight layers
  Mat* w_d = malloc((model->n_layers - 1) * sizeof(Mat));
  Mat* delta = malloc(model->n_layers * sizeof(Mat));

  for (int i = 0; i < model->n_layers; i++)
  {
    int neurons = (i == 0) ? d : (int)model->layers[i - 1].w->cols;

    // Activations and Errors: (Batch x Neurons)
    a[i].rows = batch_s;
    a[i].cols = neurons;
    a[i].data = calloc(batch_s * neurons, sizeof(float));

    z[i].rows = batch_s;
    z[i].cols = neurons;
    z[i].data = calloc(batch_s * neurons, sizeof(float));

    delta[i].rows = batch_s;
    delta[i].cols = neurons;
    delta[i].data = calloc(batch_s * neurons, sizeof(float));

    // Transposed Activations: (Neurons x Batch)
    a_T[i].rows = neurons;
    a_T[i].cols = batch_s;
    a_T[i].data = calloc(neurons * batch_s, sizeof(float));

    // Weight-related workspace (only for layers with weights)
    if (i < model->n_layers - 1) {
      w_T[i].rows = model->layers[i].w->cols;
      w_T[i].cols = model->layers[i].w->rows;
      w_T[i].data = calloc(w_T[i].rows * w_T[i].cols, sizeof(float));

      w_d[i].rows = model->layers[i].w->rows;
      w_d[i].cols = model->layers[i].w->cols;
      w_d[i].data = calloc(w_d[i].rows * w_d[i].cols, sizeof(float));
    }
  }

  Mat* Y_batch = mat_new(batch_s, c);
  Mat* softmax = mat_new(batch_s, c);

  // --- 2. TRAINING LOOPS ---
  int correct_this_batch = 0;
  for (int epoch = 0; epoch < model->max_epochs; epoch++)
  {

    for (int b = 0; b < n; b += batch_s)
    {
      correct_this_batch = 0;
      int printout = (X->rows / batch_s / 5 && b % PRINT_FREQUENCY == 0);
      // printf("%d\n",b);
      int current_batch_size = (b + batch_s < n) ? batch_s : n - b;

      // If current_batch_size becomes negative or zero, memcpy may fail or crash.
      if (current_batch_size <= 0)
        break;

      // Sync metadata for potential small final batch
      for (int i = 0; i < model->n_layers; i++)
      {
        a[i].rows = current_batch_size;
        z[i].rows = current_batch_size;
        delta[i].rows = current_batch_size;
      }
      Y_batch->rows = current_batch_size;
      softmax->rows = current_batch_size;

      // 1. Load Batch Data (Contiguous Row Copy)
      // Because examples are rows, we can copy the whole block at once
      memcpy(a[0].data, &X->data[b * d], current_batch_size * d * sizeof(float));
      memcpy(Y_batch->data, &Y_labels->data[b * c], current_batch_size * c * sizeof(float));

      // 2. Forward Pass: Z = A_prev * W + B
      for (int i = 1; i < model->n_layers; i++)
      {
        mat_dot(&a[i - 1], model->layers[i - 1].w, &z[i]);
        mat_add_bias(&z[i], model->layers[i - 1].b);
        mat_apply(&z[i], relu, &a[i]);
      }

      // 3. Stable Softmax (Row-wise for each sample in batch)
      for (int i = 0; i < a[i - 1].rows; i++)
      {
        float max_val = -INFINITY;
        int row_offset = i * a[i - 1].cols;

        for (int j = 0; j < a[i - 1].cols; j++)
        {
          if (a[model->n_layers - 1].data[row_offset + j] > max_val)
            max_val = a[model->n_layers - 1].data[row_offset + j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < a[i - 1].cols; j++)
        {
          int idx = row_offset + j;
          softmax->data[idx] = expf(a[model->n_layers - 1].data[idx] - max_val);
          sum_exp += softmax->data[idx];
        }
        for (int j = 0; j < a[i - 1].cols; j++)
        {
          softmax->data[row_offset + j] /= sum_exp;
        }
      }

      for (int i = 0; i < current_batch_size; i++)
      {
        int row_offset = i * c;
        int predicted = 0;
        float max_p = -999999999;
        int actual = 0;

        for (int j = 0; j < c; j++)
        {
          // Find predicted digit from Softmax
          if (softmax->data[row_offset + j] > max_p)
          {
            max_p = softmax->data[row_offset + j];
            predicted = j;
          }
          // Find actual digit from Y_batch
          if (Y_batch->data[row_offset + j] == 1.0f)
          {
            actual = j;
          }
        }
        if (printout && predicted == actual)
          correct_this_batch++;
      }

      // batch printout
      if (printout)
      {
        double train_acc = (double)correct_this_batch / (current_batch_size);
        printf("Epoch %2.d | Batch %5.d/%5.d | Batch Accuracy: %.2f%%\n", epoch + 1, b + batch_s, n, train_acc * 100.0f);
      }
      // 4. Backward Pass
      // Delta for output layer (Softmax - Target)
      for (int i = 0; i < current_batch_size * c; i++) {
        delta[model->n_layers - 1].data[i] = (softmax->data[i] - Y_batch->data[i]) / current_batch_size;
      }

      for (int i = model->n_layers - 1; i > 0; i--) {
        // Current layer we are processing: weights are at index i-1
        Layer* layer = &model->layers[i - 1];

        // Gradient for weights: W_grad = A_prev^T * Delta_curr
        // A_prev is a[i-1], Delta_curr is delta[i]
        mat_transpose(&a[i - 1], &a_T[i - 1]); // Note: Use index i-1 for transpose workspace

        // Wd should match dimensions of layer->w (n_in x n_out)
        Mat* Wd = &w_d[i - 1];
        mat_dot(&a_T[i - 1], &delta[i], Wd);

        // If not at the input layer, backpropagate delta to the previous layer
        if (i > 1) {
          Mat* WT = &w_T[i - 1];
          mat_transpose(layer->w, WT);
          mat_dot(&delta[i], WT, &delta[i - 1]);

          // Apply derivative of activation function
          mat_apply(&z[i - 1], relu_d, &z[i - 1]);
          mat_hadamard(&delta[i - 1], &z[i - 1], &delta[i - 1]);
        }

        // Gradient Descent Update for Weights
        mat_scale(Wd, model->lr, Wd); // current_batch_size division already handled in delta init
        mat_sub(layer->w, Wd, layer->w);

        // Bias update: sum deltas across the batch
        int layer_neurons = layer->b->cols;
        for (int j = 0; j < layer_neurons; j++) {
          float b_grad = 0;
          for (int row = 0; row < current_batch_size; row++) {
            b_grad += delta[i].data[row * layer_neurons + j];
          }
          layer->b->data[j] -= model->lr * b_grad;
        }
      }
    }
    double train_acc = calculate_accuracy(model, X, Y_labels);
    printf("========================================================\n");
    printf("Epoch %2.d | Epoch Total       | Train Accuracy: %.2f%%\n", epoch + 1, train_acc * 100.0f);
    printf("========================================================\n");
  }

  // --- 3. CLEANUP ---
  for (int i = 0; i < model->n_layers; i++)
  {
    free(a[i].data);
    free(a_T[i].data);
    free(z[i].data);
    free(delta[i].data);

    if (i < model->n_layers - 1) {
      free(w_T[i].data);
      free(w_d[i].data);
    }
  }

  free(a);
  free(a_T);
  free(z);
  free(delta);
  free(w_T);
  free(w_d);

  mat_free(Y_batch);
  mat_free(softmax);
}

double calculate_accuracy(Model* model, Mat* X, Mat* Y_one_hot)
{
  // model_predict should now return a (Samples x Classes) matrix
  Mat* predictions = model_predict(model, X);
  int correct = 0;
  int n_samples = X->rows;         // Rows are examples
  int n_classes = Y_one_hot->cols; // Columns are classes

  for (int i = 0; i < n_samples; i++)
  {
    int predicted_digit = 0;
    float max_val = -999999999999.0;
    int row_offset = i * n_classes;

    // 1. Find index of max value in the prediction row
    for (int j = 0; j < n_classes; j++)
    {
      float val = predictions->data[row_offset + j];
      if (val > max_val)
      {
        max_val = val;
        predicted_digit = j;
      }
    }

    // 2. Find index of actual digit in the one-hot row
    int actual_digit = 0;
    for (int j = 0; j < n_classes; j++)
    {
      if (Y_one_hot->data[row_offset + j] == 1.0f)
      {
        actual_digit = j;
        break;
      }
    }

    if (predicted_digit == actual_digit)
    {
      correct++;
    }
  }

  mat_free(predictions);
  return ((double)correct) / n_samples;
}

Mat* model_predict(Model* model, Mat* X)
{
  Mat* current_a = X;

  for (int i = 1; i < model->n_layers; i++)
  {
    Mat* curr_w = model->layers[i - 1].w;
    Mat* curr_b = model->layers[i - 1].b;

    // Output must be (Samples x Output_Neurons)
    Mat* next_a = mat_new(current_a->rows, curr_w->cols);

    mat_dot(current_a, curr_w, next_a); // Row-Major: (N x In) * (In x Out)
    mat_add_bias(next_a, curr_b);
    mat_apply(next_a, relu, next_a);

    if (current_a->cols != curr_w->rows) {
      printf("Dimension Mismatch at Layer %d: Input Cols (%d) != Weight Rows (%d)\n",
        i, current_a->cols, curr_w->rows);
    }

    // Free the intermediate activation matrix to prevent leaks
    if (current_a != X)
    {
      mat_free(current_a);
    }
    current_a = next_a;
  }
  return current_a;
}

void model_free(Model* model)
{
  for (int i = 0; i < model->n_layers - 1; i++)
  {
    mat_free(model->layers[i].w);
    mat_free(model->layers[i].b);
  }
  free(model->layers); // Don't forget to free the layers array itself!
  free(model);
}

Mat* one_hot_encode(Mat* labels, int num_classes)
{
  Mat* one_hot = mat_new(labels->rows, num_classes);
  for (int i = 0; i < labels->rows; i++)
  {
    one_hot->data[i * num_classes + (int)labels->data[i]] = 1;
  }
  return one_hot;
}