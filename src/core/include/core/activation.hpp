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
        inline real_number_t activation(vector_t& activations, size_t index){
            return 1 / (1 + exp(-activations[index]));
        }
        inline real_number_t derivative(vector_t& activations, size_t index){
            return activations[index] * (1 - activations[index]);
        }
    }
    namespace relu
    {
        inline real_number_t activation(vector_t& activations, size_t index){
            return std::max(real_number_t(0), activations[index]);
        }
        inline real_number_t derivative(vector_t& activations, size_t index){
            return activations[index] > 0;
        }
    } // namespace relu

    namespace softmax
    {
        real_number_t activation(vector_t& args, size_t index);
        real_number_t derivative(vector_t& activations, size_t index);
    }

    namespace silu
    {
        real_number_t activation(vector_t& args, size_t index);
        real_number_t derivative(vector_t& activations, size_t index);
    }

    namespace selu
    {
        real_number_t activation(vector_t& args, size_t index);
        real_number_t derivative(vector_t& activations, size_t index);
    }
    namespace prelu
    {
        real_number_t activation(vector_t& args, size_t index);
        real_number_t derivative(vector_t& activations, size_t index);
    }
}