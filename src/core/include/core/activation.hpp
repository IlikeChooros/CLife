#pragma once

#include <algorithm>
#include <cmath>

#include "namespaces.hpp"
#include "types.hpp"

enum class ActivationType {
    sigmoid,
    relu,
    softmax,
    silu,
    selu,
    prelu
};

class BaseActivation{
public:
    BaseActivation(ActivationType&& type = ActivationType::sigmoid): type(type) {}
    virtual double activation(double arg) = 0;
    virtual double derivative(double activation) = 0;
    ActivationType type;
};

/// One of base activation function types, standard activation function
/// it smooths out activation beetween values 0 and 1
/// 
/// f(x) = 1 / (1 + e^(-x))
class Sigmoid: public BaseActivation{
public:
    Sigmoid() : BaseActivation(ActivationType::sigmoid) {}
    double activation(double arg){
        return 1 /  (1 + exp(-arg));
    }
    double derivative(double activation){
        return activation * (1 - activation);
    }
};

/// One of base activation function types, considered the fastest
/// -> using only addition, multiplication and substraction when backpropagation process
/// goes on. Causes `dying neurons` effect, because if the activation is below 0, then 
/// it will never arise again, it dies
///
/// f(x) = if x <= 0 then: 0, else: x
class ReLu: public BaseActivation{
public:
    ReLu() : BaseActivation(ActivationType::relu) {}
    double activation(double arg){
        return std::max((double)0, arg);
    }
    double derivative(double activation){
        return activation > 0 ? 1 : 0;
    }
};

/// New - namespace approach
namespace neural_network{
    namespace sigmoid{
        inline vector_t activation(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = 1 / (1 + exp(-activations[i]));
            }
            return result;
        }
        inline vector_t derivative(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = activations[i] * (1 - activations[i]);
            }
            return result;
        }
    }
    namespace relu
    {
        inline vector_t activation(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = std::max((real_number_t)0, activations[i]);
            }
            return result;
        }
        inline vector_t derivative(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = activations[i] > 0 ? 1 : 0;
            }
            return result;
        }
    } // namespace relu

    namespace softmax
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }

    namespace silu
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }

    namespace selu
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }
    namespace prelu
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }
}