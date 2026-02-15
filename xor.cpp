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

float cost( Xor m) {
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
    std::cout << "\n=== XOR Neural Network Parameters ===\n";
    
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
    
    std::cout << "\n=====================================\n";
}

const float eps = 1e-4f;  

Xor finite_diff(Xor m) {
    Xor g;
    float c = cost(m);  

    float saved;

    // or_w1
    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    // or_w2
    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    // or_b
    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    // and_w1
    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    // and_w2
    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    // and_b
    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    // nand_w1
    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    // nand_w2
    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    // nand_b
    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    return g;
}

Xor learn(Xor m, Xor g, float rate) {
    m.or_w1   -= rate * g.or_w1;
    m.or_w2   -= rate * g.or_w2;
    m.or_b    -= rate * g.or_b;

    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b  -= rate * g.nand_b;

    m.and_w1  -= rate * g.and_w1;
    m.and_w2  -= rate * g.and_w2;
    m.and_b   -= rate * g.and_b;

    return m;
}
int main() {
    std::srand(std::time(nullptr));

    Xor m = rand_xor();
    printf("Initial cost = %f\n", cost(m));

    const float rate = 0.05f;           
    const int epochs = 900000;           

    for (int i = 0; i < epochs; ++i) {
        Xor grad = finite_diff(m);
        m = learn(m, grad, rate);

        if (i % 1000 == 0) {
            printf("Epoch %d | Cost = %f\n", i, cost(m));
        }
    }

    printf("Final cost = %f\n", cost(m));
    print_xor(m);

    printf("\n--- XOR Results ---\n");
    printf("1 XOR 1 = %f\n", forward(m, 1, 1));
    printf("1 XOR 0 = %f\n", forward(m, 1, 0));
    printf("0 XOR 1 = %f\n", forward(m, 0, 1));
    printf("0 XOR 0 = %f\n", forward(m, 0, 0));
}
