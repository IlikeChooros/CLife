#pragma once

#include <algorithm>
#include <cmath>

enum class ActivationType {
    sigmoid,
    relu,
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
        inline double activation(double arg){
            // auto ret = 1 / (1 + exp(-arg));
            // if (std::isnan(ret)){
            //     while(1){

            //     }
            // }
            return 1 / (1 + exp(-arg));
        }
        inline double derivative(double activation){
            return activation * (1 - activation);
        }
    }
    namespace relu
    {
        inline double activation(double arg){
            return std::max(double(0), arg);
        }
        inline double derivative(double activation){
            return activation > 0;
        }
    } // namespace relu
    
    typedef double(*function_pointer)(double);

    inline function_pointer matchType(ActivationType&& type, function_pointer on_sigmoid, function_pointer on_relu){
        switch (type)
        {
        case ActivationType::sigmoid:
            return on_sigmoid;
        case ActivationType::relu:
            return on_relu;
        default:
            return nullptr;
        }
    }

    inline function_pointer matchActivation(ActivationType&& type){
        return matchType(std::forward<ActivationType>(type), sigmoid::activation, relu::activation);
    }

    inline function_pointer matchDerivative(ActivationType&& type){
        return matchType(std::forward<ActivationType>(type), sigmoid::derivative, relu::derivative);
    }
}