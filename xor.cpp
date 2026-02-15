#include <math.h>
#include <atomic>
#include <iostream>

typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float and_w1;
    float and_w2;
    float and_b;

    float nand_w1;
    float nand_w2;
    float nand_b;
} Xor;

float sigmoid(float x){
    return 1.f/(1.f+ expf(-x));
}

float forward(Xor m , float x1 , float x2){

    float a = sigmoid(x1*m.or_w1 + x2*m.or_w2 + m.or_b);
    float b = sigmoid(x1*m.nand_w1 + x2*m.nand_w2 + m.nand_b);

    return (sigmoid(a*m.and_w1 + b*m.and_w2 + m.and_b));
}


float xor_train[][3] = {
    {1 , 1 , 0},
    {1 , 0 , 1},
    {0 , 1 , 1},
    {0 , 0 , 0},
};

#define train_size_XOR 4

float costf_AND( Xor m) {
    float result = 0.0f;
    for (size_t i = 0; i < train_size_XOR; i++) {
        float x1 = xor_train[i][0];
        float x2 = xor_train[i][1];
        float true_val = xor_train[i][2];
        float y = forward(m , x1 , x2);
        float err = y - true_val;
        result += err * err;
    }
    return result / train_size_XOR; 
}


float randfloat() {
    return (float)rand() / (float)RAND_MAX;
}

Xor rand_xor(){
    Xor m;
    m.or_w1 = randfloat();
    m.or_w2 = randfloat();
    m.or_b = randfloat();

    m.and_w1 = randfloat();
    m.and_w2 = randfloat();
    m.and_b = randfloat();

    m.nand_w1 = randfloat();
    m.nand_w2 = randfloat();
    m.nand_b = randfloat();

    return m;
}

void print_xor(const Xor& m) {
    std::cout << "=== XOR Neural Network Parameters ===\n";
    
    std::cout << "OR gate:\n";
    std::cout << "  w1: " << m.or_w1 << "\n";
    std::cout << "  w2: " << m.or_w2 << "\n";
    std::cout << "  b : " << m.or_b  << "\n\n";
    
    std::cout << "AND gate:\n";
    std::cout << "  w1: " << m.and_w1 << "\n";
    std::cout << "  w2: " << m.and_w2 << "\n";
    std::cout << "  b : " << m.and_b  << "\n\n";
    
    std::cout << "NAND gate:\n";
    std::cout << "  w1: " << m.nand_w1 << "\n";
    std::cout << "  w2: " << m.nand_w2 << "\n";
    std::cout << "  b : " << m.nand_b  << "\n";
    
    std::cout << "=====================================\n";
}

int main() {
    //std::srand(std::time(nullptr));  
    Xor m = rand_xor();
    print_xor(m);
    return 0;
}
