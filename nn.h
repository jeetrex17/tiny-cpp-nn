#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <random>
#include <algorithm>

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif

namespace nn {

    enum class Activation {
        Sigmoid,
        Relu, 
        Tanh,
        Sin
    } ;
    #ifndef NN_ACT
    #define NN_ACT nn::Activation::Sigmoid
    #endif 

    inline float Sigmoid (float x) { return (1/(1+exp(-x)));}
    inline float Relu (float x) { return std::max(0.0f,x);}
    inline float Tanh (float x) { return ((exp(x) - exp(-x))/(exp(x) + exp(-x)));}
    inline float Sin (float x) { return std::sin(x);}


    // Activation Function
    inline float Actf(float x , Activation act = NN_ACT){
        switch (act) {
            case Activation::Sigmoid: return Sigmoid(x);
            case Activation::Relu: return Relu(x);
            case Activation::Tanh: return Tanh(x);
            case Activation::Sin: return Sin(x);
        }
        assert(false && "unreachable");
        return 0.0f;
    }

    // Derivative of the Activation Function
    inline float Dactf(float y , Activation dact = NN_ACT){
        switch (dact) {
            case Activation::Sigmoid: return  y * (1.0f - y);
            case Activation::Relu:  return y >= 0 ? 1.0f : NN_RELU_PARAM;
            case Activation::Tanh:  return 1.0f - y * y;
            case Activation::Sin: return std::cos(std::asin(y)); 
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

        float& operator()(size_t i , size_t j){
            assert(i<this->rows && j< this->cols);
            return this->data[i*cols + j];
        }

        // const one for reading only;
        const float& operator()(size_t i, size_t j) const {
            assert(i < rows && j < cols);
            return data[i * cols + j];
        }

        void fill(float x){
            std::fill(data.begin() , data.end() , x);
        }

        void randomize(float low , float high){
            float x = rand_float(low, high);
            std::fill(data.begin() , data.end() , x);
        }

        float apply_activation(Activation act){
            for(auto &d : data){
                d = Actf(d, act);
            }
        }

        Matrix& operator+=(const Matrix& other){
            assert(other.cols == cols && rows == other.rows);
            for(size_t i = 0; i < data.size() ; i++){
                data[i] = data[i]+other.data[i];
            }

            return *this;
        }

        static Matrix dot(const Matrix& a , const Matrix& b){
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
        
        Matrix slice_row(size_t row_idx , size_t start_col , size_t num_cols){
            assert(row_idx < rows);
            assert(start_col + num_cols <= cols);
            Matrix m(1 , num_cols );
            for(size_t k ; k < num_cols ; k++){
                m(0,k) = (*this)(row_idx , start_col + k);
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
        std::vector<size_t> arch; // Architecture it stores number of neuros per layer
        std::vector<Matrix> ws;   // Weights
        std::vector<Matrix> bs;   // Biases
        std::vector<Matrix> as;   // Activations

        NeuralNetwork (const std::vector<size_t>& architecture) : arch(architecture) {
            assert(arch.size() > 0);

            as.emplace_back(1 , arch[0]); // input layer for example if arch is {2 , 3 , 1} then input matix should be 1x2
            for (size_t i = 1; i < arch.size(); ++i) {
                ws.emplace_back(arch[i-1], arch[i]);
                bs.emplace_back(1, arch[i]);
                as.emplace_back(1, arch[i]);
            }
        }

        Matrix& get_input() {
            return as.front();
        }
        Matrix& get_output() {
            return as.back();
        }
        
        const Matrix& get_output() const {
             return as.back();
        }
        void zero() {
            for(auto a : as) { a.fill(0.0f) ;}
            for(auto w : ws) { w.fill(0.0f);}
            for(auto b : bs) { b.fill(0.0f);}
        }

        void randomize(float low , float high) {
            for(auto a : as) { a.randomize(low, high);}
            for(auto w : ws) { w.randomize(low, high);}
            for(auto b : bs) { b.randomize(low, high);}
        }
        void print(const std::string& name = "nn") const {
            std::cout << name << " = [\n";
            for (size_t i = 0; i < ws.size(); ++i) {
                ws[i].print("ws" + std::to_string(i), 4);
                bs[i].print("bs" + std::to_string(i), 4);
            }
            std::cout << "]\n";
        }
        // TODO 19: Write 'forward()'.
        // The main feed-forward loop. Loop through the layers:
        // 1. dot product the current activation with the weight.
        // 2. add the bias.
        // 3. apply the activation function.
        // Save the result in the next activation layer.

        // TODO 20: Write 'cost(const Matrix& t)'.
        // Loop through the training rows, run forward(), and calculate the Mean Squared Error.

        // TODO 21: Write 'backprop(const Matrix& t)'.
        // Create a new 'gradient' NeuralNetwork. Run forward(). Calculate output errors. 
        // Loop backwards through layers to calculate weight/bias gradients using your 'dactf' derivative function.
        // Return the gradient NeuralNetwork.

        // TODO 22: Write 'learn(const NeuralNetwork& g, float rate)'.
        // Loop through your weights/biases and subtract (rate * gradient_weights).
    };

    // ==========================================
    // 4. BATCHING STRUCT
    // ==========================================
    
    struct Batch {
        // TODO 23: Define 'size_t begin', 'float cost', and 'bool finished'.

        // TODO 24: Write 'process(size_t batch_size, NeuralNetwork& nn, const Matrix& t, float rate)'.
        // Slice the target matrix into a batch, run backprop, run learn, update cost and the begin index.
    };

} // namespace nn
