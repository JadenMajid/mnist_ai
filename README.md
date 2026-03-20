# MNIST AI Implementation in C

This project is a high-performance, custom-built feedforward neural network implemented entirely in C. It trains on the MNIST dataset to perform handwritten digit classification without relying on high-level machine learning libraries like PyTorch or TensorFlow.

## Core Features

- **Custom Neural Network Architecture**: Supports dynamic multi-layer perceptron (MLP) topologies configurable at runtime.
- **Optimized Linear Algebra**: Matrix operations are implemented from scratch and optimized using Apple's Accelerate framework for hardware-accelerated computations.
- **Stable Training**: Implements mini-batch gradient descent (SGD) with numerically stable Softmax and Cross-Entropy loss.
- **Model Persistence**: Includes a custom binary serialization format for saving, loading, and resuming model training states.
- **Memory Management**: Careful memory allocation and cleanup to prevent memory leaks during extended training loops.

## Technical Details

- **Language**: C
- **Compiler Options**: `-O3 -mcpu=apple-m1 -ffast-math -flto` for maximum performance on macOS architectures.
- **Forward Pass**: Uses the ReLU activation function for hidden layers and Softmax for the output layer.
- **Backward Pass**: Derives and applies analytical gradients through backpropagation.

## Building the Project

Ensure you have `gcc` and `make` installed. The project currently targets macOS utilizing the Accelerate framework (`-framework Accelerate`).

```sh
make clean
make
```

The compiled binary will be placed at `out/main`.

## Usage Instructions

The `main` executable accepts several options to control the network topology, hyperparameters, and persistence.

```
Usage: ./out/main [OPTIONS]
Options:
  -t, --topology=STR  Layer sizes, e.g., '784,128,64,10'
  -e, --epochs=INT    Number of training epochs (default: 10)
  -l, --lr=FLOAT      Learning rate (default: 0.1)
  -b, --batch=INT     Batch size (default: 256)
  -s, --save          Save model after training
  -f, --load=NAME     Load model from data/NAME
  -h, --help          Display this help message
```

### Example Configurations

**Train a new model:**
Trains a default topology (784 -> 128 -> 64 -> 10) for 15 epochs with a learning rate of 0.05.
```sh
./out/main -e 15 -l 0.05
```

**Train a custom topology and save it:**
Trains a wider network and saves the learned weights and biases to the `data/` directory.
```sh
./out/main -t "784,256,128,10" -s
```

**Evaluate a saved model:**
Loads a model previously saved (e.g., under timestamp `1773990297`) and evaluates its test accuracy without further training.
```sh
./out/main -f 1773990297 -e 0
```

## Dataset

The implementation expects the MNIST dataset files natively encoded in IDX format, located in the `archive/ua_tr/` directory relative to the binary execution path.

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`
