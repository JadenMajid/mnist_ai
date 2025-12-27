#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include "linalg.h"
#include "model.h"
#include "mnist_loader.h"
#include "fs.h"

// Helper to parse "784,128,10" into a Mat* topology
Mat* parse_topology(char *optarg) {
    int count = 1;
    for (int i = 0; optarg[i]; i++) if (optarg[i] == ',') count++;

    Mat *topo = mat_new(count, 1);
    char *copy = strdup(optarg);
    char *token = strtok(copy, ",");
    int i = 0;
    while (token != NULL) {
        topo->data[i++] = (float)atof(token);
        token = strtok(NULL, ",");
    }
    free(copy);
    return topo;
}

void print_usage(char *prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -t, --topology=STR  Layer sizes, e.g., '784,128,64,10'\n");
    printf("  -e, --epochs=INT    Number of training epochs (default: 10)\n");
    printf("  -l, --lr=FLOAT      Learning rate (default: 0.1)\n");
    printf("  -b, --batch=INT     Batch size (default: 256)\n");
    printf("  -s, --save          Save model after training\n");
    printf("  -f, --load=NAME     Load model from data/NAME\n");
    printf("  -h, --help          Display this help message\n");
}

int main(int argc, char **argv) {
    int epochs = 10;
    float learning_rate = 0.1f;
    int batch_size = 256;
    int should_save = 0;
    char *load_path = NULL;
    Mat *topology = NULL;

    static struct option long_options[] = {
        {"topology", required_argument, 0, 't'},
        {"epochs",   required_argument, 0, 'e'},
        {"lr",       required_argument, 0, 'l'},
        {"batch",    required_argument, 0, 'b'},
        {"save",     no_argument,       0, 's'},
        {"load",     required_argument, 0, 'f'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:e:l:b:sf:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 't': topology = parse_topology(optarg); break;
            case 'e': epochs = atoi(optarg); break;
            case 'l': learning_rate = atof(optarg); break;
            case 'b': batch_size = atoi(optarg); break;
            case 's': should_save = 1; break;
            case 'f': load_path = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
            default: return 1;
        }
    }

    if (topology) {
    printf("Custom topology detected: ");
    for (int i = 0; i < topology->rows; i++) {
        printf("%d%s", (int)topology->data[i], (i == topology->rows - 1) ? "" : " -> ");
    }
    printf("\n");
}

    srand(time(NULL));

    Model *model = NULL;
    if (load_path) {
        printf("Loading model from data/%s...\n", load_path);
        model = load_model(load_path);
    } else {
        // Use default topology if none provided via -t
        if (!topology) {
            topology = mat_new(4, 1);
            topology->data[0] = 784; 
            topology->data[1] = 128; 
            topology->data[2] = 64;  
            topology->data[3] = 10;
        }
        mat_print(topology);
        model = model_new(topology, epochs, learning_rate);
    }

    if (!model) { fprintf(stderr, "Error: Model initialization failed.\n"); return 1; }

    // 2. Load MNIST Data
    printf("Loading MNIST dataset...\n");
    Mat *X_train = read_mnist_images("archive/train-images.idx3-ubyte");
    Mat *Y_train = read_mnist_labels("archive/train-labels.idx1-ubyte");
    Mat *X_test = read_mnist_images("archive/test-images.idx1-ubyte");
    Mat *Y_test_raw = read_mnist_labels("archive/test-labels.idx1-ubyte");

    if (!X_train || !Y_train) return 1;

    // 3. Training
    printf("Configuration: LR=%.4f, Epochs=%d, Batch=%d\n", learning_rate, epochs, batch_size);
    clock_t start = clock();
    model_train(model, X_train, Y_train, batch_size);
    clock_t end = clock();

    printf("\nTraining complete in %.2f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 4. Evaluation
    float acc = calculate_accuracy(model, X_test, Y_test_raw);
    printf("Final Test Accuracy: %.2f%%\n", acc * 100.0f);

    // 5. Saving
    if (should_save) {
        char timestamp[64];
        snprintf(timestamp, sizeof(timestamp), "%ld", (long)time(NULL));
        save_model(timestamp, model);
        printf("Model saved to data/%s\n", timestamp);
    }

    // 6. Cleanup
    if (topology) mat_free(topology);
    mat_free(X_train); mat_free(Y_train);
    mat_free(X_test);  mat_free(Y_test_raw);
    model_free(model);

    return 0;
}