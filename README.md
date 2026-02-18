# tiny-cpp-nn

A tiny, header-only neural network library in modern C++ built for learning.


## Features

- Header-only: just include `nn.h`
- Minimal `Matrix` type (2D float matrix + dot product + row slicing)
- Activations: Sigmoid, ReLU, Tanh, Sin
- Forward pass
- Mean Squared Error (MSE) cost over a small dataset matrix
- Backpropagation (computes weight/bias gradients)
- SGD update step (`learn`)
- Simple mini-batching helper (`nn::Batch`)
- Zero external dependencies

## Use cases

- for solve simple problems like XOR or tiny regression experiments
- Quick experiments with activations and learning rates

## Requirements

- C++20 compiler 
- Works with clang++ and g++

## Repo layout

- `nn.h` — the header-only library
- `demo/3x.cpp` — learns `y = 3x` (tiny regression demo)
- `demo/xor_nn.cpp` — learns XOR using backprop + mini-batching
- `demo/old_*.cpp` — older/experimental finite-difference prototypes

## Build & run

```bash
# clang++
clang++ -std=c++20 -O2 demo/3x.cpp -o demo_3x && ./demo_3x
clang++ -std=c++20 -O2 demo/xor_nn.cpp -o demo_xor && ./demo_xor

# g++
g++ -std=c++20 -O2 demo/3x.cpp -o demo_3x && ./demo_3x
g++ -std=c++20 -O2 demo/xor_nn.cpp -o demo_xor && ./demo_xor
```


## Library overview

### Core types

- `nn::Matrix`
  - Stores `rows`, `cols`, and `std::vector<float> data`
  - Key helpers: `dot(a, b)`, `slice_row(...)`, `apply_activation(...)`
- `nn::NeuralNetwork`
  - Create with an architecture like `{2, 4, 1}` (input → hidden → output)
  - Key methods: `randomize(low, high)`, `forward()`, `cost(train)`, `backprop(train)`, `learn(gradients, rate)`
- `nn::Batch`
  - Mini-batch stepping helper: repeatedly call `process(...)` until `finished == true`

### Training data format

Training uses a single matrix `t` where each row is:

```
[ input_0, input_1, ... , target_0, target_1, ... ]
```

So `t.cols == input_dim + output_dim`. For XOR (2 inputs, 1 output), each row has 3 columns.

### Minimal usage example

```cpp
#include "nn.h"
#include <vector>

int main() {
  std::vector<size_t> arch = {2, 4, 1};
  nn::NeuralNetwork net(arch);
  net.randomize(-1.0f, 1.0f);

  net.get_input()(0, 0) = 0.0f;
  net.get_input()(0, 1) = 1.0f;
  net.forward(); // uses the default activation (see NN_ACT)

  float y = net.get_output()(0, 0);
  (void)y;
}
```

### Single training step

```cpp
// train: rows = samples, cols = input_dim + output_dim
nn::NeuralNetwork grad = net.backprop(train);
net.learn(grad, /*learning_rate=*/0.1f);
```

## Configuration (macros)

These are compile-time switches (define them before including `nn.h`, or pass `-D...` to the compiler):

- `NN_ACT`
  - Sets the default activation used by `forward()` and training/backprop.
  - Example: `-DNN_ACT=nn::Activation::Relu`
- `NN_RELU_PARAM`
  - Used as the “leaky” slope in the ReLU derivative branch.
  - Example: `-DNN_RELU_PARAM=0.01f`
- `NN_BACKPROP_TRADITIONAL`
  - Toggles an alternate backprop scaling path used in the header (see `demo/xor_nn.cpp` for how it’s enabled).

## TODO

- [ ] Scalar multiplication for `Matrix`
- [ ] Multi-threaded dot product
- [ ] Matrix transpose / inverse 
