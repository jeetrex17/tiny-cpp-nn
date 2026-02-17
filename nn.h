#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif

namespace nn {

enum class Activation { Sigmoid, Relu, Tanh, Sin };
#ifndef NN_ACT
#define NN_ACT nn::Activation::Sigmoid
#endif

inline float Sigmoid(float x) { return (1 / (1 + exp(-x))); }
inline float Relu(float x) { return std::max(0.0f, x); }
inline float Tanh(float x) { return ((exp(x) - exp(-x)) / (exp(x) + exp(-x))); }
inline float Sin(float x) { return std::sin(x); }

// Activation Function
inline float Actf(float x, Activation act = NN_ACT) {
  switch (act) {
    case Activation::Sigmoid:
      return Sigmoid(x);
    case Activation::Relu:
      return Relu(x);
    case Activation::Tanh:
      return Tanh(x);
    case Activation::Sin:
      return Sin(x);
  }
  assert(false && "unreachable");
  return 0.0f;
}

// Derivative of the Activation Function
inline float Dactf(float y, Activation dact = NN_ACT) {
  switch (dact) {
    case Activation::Sigmoid:
      return y * (1.0f - y);
    case Activation::Relu:
      return y >= 0 ? 1.0f : NN_RELU_PARAM;
    case Activation::Tanh:
      return 1.0f - y * y;
    case Activation::Sin:
      return std::cos(std::asin(y));
  }
  assert(false && "Unreachable");
  return 0.0f;
}

inline float rand_float(float low, float high) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(low, high);
  return dist(gen);
}

class Matrix {
 public:
  size_t rows;
  size_t cols;
  std::vector<float> data;

  Matrix(size_t r = 0, size_t c = 0, float d = 0.0f)
      : rows(r), cols(c), data(r * c, d) {}

  float& operator()(size_t i, size_t j) {
    assert(i < this->rows && j < this->cols);
    return this->data[i * cols + j];
  }

  // const one for reading only;
  const float& operator()(size_t i, size_t j) const {
    assert(i < rows && j < cols);
    return data[i * cols + j];
  }

  void fill(float x) { std::fill(data.begin(), data.end(), x); }

  void randomize(float low, float high) {
    float x = rand_float(low, high);
    std::fill(data.begin(), data.end(), x);
  }

  void apply_activation(Activation act) {
    for (auto& d : data) {
      d = Actf(d, act);
    }
  }

  Matrix& operator+=(const Matrix& other) {
    assert(other.cols == cols && rows == other.rows);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = data[i] + other.data[i];
    }

    return *this;
  }

  static Matrix dot(const Matrix& a, const Matrix& b) {
    assert(a.cols == b.rows);
    Matrix dst(a.rows, b.cols, 0.0f);
    for (size_t i = 0; i < dst.rows; ++i) {
      for (size_t j = 0; j < dst.cols; ++j) {
        for (size_t k = 0; k < a.cols; ++k) {
          dst(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return dst;
  }

  // row index , start col idx and how many next cols u want is that nums_cols
  Matrix slice_row(size_t row_idx, size_t start_col, size_t num_cols) const {
    assert(row_idx < rows);
    assert(start_col + num_cols <= cols);
    Matrix m(1, num_cols);
    for (size_t k = 0; k < num_cols; k++) {
      m(0, k) = (*this)(row_idx, start_col + k);
    }
    return m;
  }

  void print(const std::string& name, size_t padding = 0) const {
    std::string pad(padding, ' ');
    std::cout << pad << name << " = [\n";
    for (size_t i = 0; i < rows; ++i) {
      std::cout << pad << "    ";
      for (size_t j = 0; j < cols; ++j) {
        std::cout << (*this)(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << pad << "]\n";
  }
};

class NeuralNetwork {
 public:
  std::vector<size_t>
      arch;                // Architecture it stores number of neuros per layer
  std::vector<Matrix> ws;  // Weights
  std::vector<Matrix> bs;  // Biases
  std::vector<Matrix> as;  // Activations

  NeuralNetwork(const std::vector<size_t>& architecture) : arch(architecture) {
    assert(arch.size() > 0);

    as.emplace_back(1, arch[0]);  // input layer for example if arch is {2 , 3 ,
                                  // 1} then input matix should be 1x2
    for (size_t i = 1; i < arch.size(); ++i) {
      ws.emplace_back(arch[i - 1], arch[i]);
      bs.emplace_back(1, arch[i]);
      as.emplace_back(1, arch[i]);
    }
  }

  Matrix& get_input() { return as.front(); }
  Matrix& get_output() { return as.back(); }

  const Matrix& get_output() const { return as.back(); }
void zero() {
    for (auto& a : as) {
      a.fill(0.0f);
    }
    for (auto& w : ws) {
      w.fill(0.0f);
    }
    for (auto& b : bs) {
      b.fill(0.0f);
    }
  }

  void randomize(float low, float high) {
    for (auto& a : as) {
      a.randomize(low, high);
    }
    for (auto& w : ws) {
      w.randomize(low, high);
    }
    for (auto& b : bs) {
      b.randomize(low, high);
    }
  }  void print(const std::string& name = "nn") const {
    std::cout << name << " = [\n";
    for (size_t i = 0; i < ws.size(); ++i) {
      ws[i].print("ws" + std::to_string(i), 4);
      bs[i].print("bs" + std::to_string(i), 4);
    }
    std::cout << "]\n";
  }

  void forward(Activation act = NN_ACT) {
    for (size_t i = 0; i < ws.size(); i++) {
      as[i + 1] = Matrix::dot(as[i], ws[i]);  // matrix multiplicaton of weight and as
      as[i + 1] += bs[i];
      as[i + 1].apply_activation(act);
    }
  }

  float cost(const Matrix& t) {
    assert(get_input().cols + get_output().cols == t.cols);
    float c = 0.0f;
    size_t n = t.rows;
    for (size_t i = 0; i < n; i++) {
      // we need to get output the true value
      Matrix inputs = t.slice_row(i, 0, get_input().cols);
      Matrix true_vals = t.slice_row(i, get_input().cols, get_output().cols);

      get_input() = inputs;
      forward();

      for (size_t j = 0; j < true_vals.cols; ++j) {
        float d = get_output()(0, j) - true_vals(0, j);
        c += d * d;
      }
    }
    return c / n;
  }

  NeuralNetwork backprop(const Matrix& t) {
    size_t n = t.rows;
    assert(get_input().cols + get_output().cols == t.cols);

    NeuralNetwork g(arch);
    g.zero();

    for (size_t i = 0; i < n; ++i) {
      Matrix in = t.slice_row(i, 0, get_input().cols);

      Matrix out = t.slice_row(i, get_input().cols, get_output().cols);

      get_input() = in;

      forward();

      for (auto& a : g.as) {
        a.fill(0.0f);
      }

      for (size_t j = 0; j < out.cols; ++j) {
#ifdef NN_BACKPROP_TRADITIONAL
        g.get_output()(0, j) = 2.0f * (get_output()(0, j) - out(0, j));
#else
        g.get_output()(0, j) = get_output()(0, j) - out(0, j);
#endif
      }

#ifdef NN_BACKPROP_TRADITIONAL
      float s = 1.0f;
#else
      float s = 2.0f;
#endif

      for (size_t l = arch.size() - 1; l > 0; --l) {
        for (size_t j = 0; j < as[l].cols; ++j) {
          float a = as[l](0, j);
          float da = g.as[l](0, j);
          float qa = Dactf(a, NN_ACT);

          g.bs[l - 1](0, j) += s * da * qa;

          for (size_t k = 0; k < as[l - 1].cols; ++k) {
            float pa = as[l - 1](0, k);
            float w = ws[l - 1](k, j);

            g.ws[l - 1](k, j) += s * da * qa * pa;
            g.as[l - 1](0, k) += s * da * qa * w;
          }
        }
      }
    }

    for (size_t i = 0; i < g.ws.size(); ++i) {
      for (size_t j = 0; j < g.ws[i].data.size(); ++j) {
        g.ws[i].data[j] /= n;
      }
      for (size_t j = 0; j < g.bs[i].data.size(); ++j) {
        g.bs[i].data[j] /= n;
      }
    }

    return g;
  }
  void learn(const NeuralNetwork& g, float rate) {
    for (size_t i = 0; i < ws.size(); ++i) {
      for (size_t j = 0; j < ws[i].data.size(); ++j) {
        ws[i].data[j] -= rate * g.ws[i].data[j];
      }
      for (size_t j = 0; j < bs[i].data.size(); ++j) {
        bs[i].data[j] -= rate * g.bs[i].data[j];
      }
    }
  }
};

struct Batch {
  size_t begin = 0;
  float cost = 0.0f;
  bool finished = false;

  void process(size_t batch_size, NeuralNetwork& nn, const Matrix& t,
               float rate) {
    if (finished) {
      finished = false;
      begin = 0;
      cost = 0.0f;
    }

    size_t size = batch_size;
    // for when batch_size gets greater than number of samples left
    if (begin + batch_size >= t.rows) {
      size = t.rows - begin;
    }

    // number of cols remain same coz u know inputs and outpus but we like make
    // small batchs of rows
    Matrix batch_t(size, t.cols);
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < t.cols; ++j) {
        batch_t(i, j) = t(begin + i, j);
      }
    }

    NeuralNetwork g = nn.backprop(batch_t);
    nn.learn(g, rate);
    cost += nn.cost(batch_t);
    begin += batch_size;

    if (begin >= t.rows) {
      size_t batch_count = (t.rows + batch_size - 1) / batch_size;
      cost /= batch_count;
      finished = true;
    }
  }
};

}  // namespace nn
