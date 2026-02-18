#define NN_ACT nn::Activation::Relu 
#include "nn.h"
#include <iostream>
#include <vector>

int main() {
    nn::Matrix train(6, 2, 0.0f);
    for (size_t i = 0; i < 6; ++i) {
        train(i, 0) = static_cast<float>(i + 1);       
        train(i, 1) = static_cast<float>(i + 1) * 3;   
    }
    train.print("training_data");

    std::vector<size_t> arch = {1, 1}; 
    nn::NeuralNetwork nn(arch);
    
    nn.randomize(-1.0f, 1.0f);

    std::cout << "\n--- Before training ---\n";
    std::cout << "Initial weight: " << nn.ws[0](0,0) << "\n";
    std::cout << "Initial bias:   " << nn.bs[0](0,0) << "\n";
    std::cout << "Initial cost:   " << nn.cost(train) << "\n\n";

    size_t epochs = 1000;
    float learning_rate = 0.005f; 

    for (size_t i = 0; i < epochs; ++i) {
        nn::NeuralNetwork gradients = nn.backprop(train);
        
        nn.learn(gradients, learning_rate);

        if (i % 100 == 0) {
            std::cout << "Epoch " << i << " | Cost: " << nn.cost(train) << "\n";
        }
    }

    std::cout << "\n--- After training ---\n";
    std::cout << "Current weight (aiming for 3.0): " << nn.ws[0](0,0) << "\n";
    std::cout << "Current bias   (aiming for 0.0): " << nn.bs[0](0,0) << "\n";

    std::cout << "\n--- Testing Predictions ---\n";
    
    nn.get_input()(0, 0) = 1.0f; 
    nn.forward(); 
    std::cout << "Prediction for x=1: " << nn.get_output()(0, 0) << " (Expected: 3)\n";

    nn.get_input()(0, 0) = 7.0f; 
    nn.forward(); 
    std::cout << "Prediction for x=7: " << nn.get_output()(0, 0) << " (Expected: 21)\n";

    return 0;
}
