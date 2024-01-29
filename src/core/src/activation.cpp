#include <core/activation.hpp>


START_NAMESPACE_NEURAL_NETWORK


namespace softmax{
    double activation(vecdouble& activations, size_t index){
        std::vector<double> softmax(activations.size());
        double sum = 0;
        for (size_t i = 0; i < activations.size(); i++){
            softmax[i] = exp(activations[i]);
            sum += softmax[i];
        }
        return softmax[index] / sum;
    }

    double derivative(vecdouble& activations, size_t index){
        return activations[index] * (1 - activations[index]);
    }
}



END_NAMESPACE