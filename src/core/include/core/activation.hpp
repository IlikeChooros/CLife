#pragma once

#include <algorithm>
#include <cmath>

#include "namespaces.hpp"

enum class ActivationType {
    sigmoid,
    relu,
    softmax
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
    using vecdouble = std::vector<double>;
    namespace sigmoid{
        inline double activation(vecdouble& activations, size_t index){
            return 1 / (1 + exp(-activations[index]));
        }
        inline double derivative(vecdouble& activations, size_t index){
            return activations[index] * (1 - activations[index]);
        }
    }
    namespace relu
    {
        inline double activation(vecdouble& activations, size_t index){
            return std::max(double(0), activations[index]);
        }
        inline double derivative(vecdouble& activations, size_t index){
            return activations[index] > 0;
        }
    } // namespace relu

    namespace softmax
    {
        double activation(vecdouble& args, size_t index);
        double derivative(vecdouble& activations, size_t index);
    }
}