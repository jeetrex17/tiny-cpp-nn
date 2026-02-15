#include <cfloat>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <ctime>

// 1.42 MIN 

int traing_data_2x[][2] = {
    {1, 3},
    {2, 6},
    {3, 9},
    {4, 12},
    {5, 15},
    {6, 18},
};

int traing_data_AND[][3] = {
    {1, 1 , 1},
    {0, 0, 0},
    {0, 1 , 0},
    {1, 0 , 0},
};
#define train_size sizeof(traing_data_2x) / sizeof(traing_data_2x[0])
#define train_size_AND sizeof(traing_data_AND) / sizeof(traing_data_AND[0])

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float randfloat() {
    return (float)rand() / (float)RAND_MAX;
}


// y = a + x(W1);

// float costf(float w1 , float b) 
// {
//     float result = 0.0f;
//     for( size_t i = 0; i < train_size ; i++) {
//         float x1 = traing_data[i][0];
//         float y = b + x1*w1;
//         float true_value = traing_data[i][1];
//         float err = y- true_value;
//         result = result + err*err;
//     }
//     return result/train_size;
// }

float costf_AND(float w1, float w2, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_size_AND; i++) {
        float x1 = traing_data_AND[i][0];
        float x2 = traing_data_AND[i][1];
        float true_val = traing_data_AND[i][2];

        float x = x1 * w1 + x2 * w2 + b;
        float y = sigmoid(x);
        float err = y - true_val;
        result += err * err;
    }
    return result / train_size_AND; 
}

int main() {
    srand(1);

    float w1 = randfloat();
    float w2 = randfloat();
    float b = randfloat();

    float alpha = 1e-1;
    float eplision = 1e-3;

    for(size_t i = 0 ; i < 1000*3000 ; i++){
        float c = costf_AND(w1,w2, b);

        float dw1 = (costf_AND(w1 + eplision, w2, b) - c)/eplision ;
        float dw2 = (costf_AND(w1 , w2 + eplision, b) - c)/eplision ;
        float db = (costf_AND(w1,w2, b + eplision) - c)/eplision;
        w1 = w1 - alpha * dw1;
        w2 = w2 - alpha * dw2;
        b = b - alpha * db;
        std::cout << "w1 : "<<w1  << "|| w2 : "<<w2 << "| b : "<< b << " || cost : " << c << std::endl;
    }

    std::cout << "--- RESULTS ---" << std::endl;
    for(size_t i = 0; i < 4 ; i++){
        int x1 = (float)traing_data_AND[i][0];
        int x2 = (float)traing_data_AND[i][1];
        std::cout << x1 << " || " << x2 << " || " << sigmoid(x1*w1 + b + x2*w2) << std::endl;
    }
    return 0;
}
