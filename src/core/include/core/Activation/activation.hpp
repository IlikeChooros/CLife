#pragma once

#include <algorithm>

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
        constexpr float c1 = 0.03138777F;
        constexpr float c2 = 0.276281267F;
        constexpr float c_log2f = 1.442695022F;
        arg *= c_log2f*0.5;
        int intPart = (int)arg;
        float x = (arg - intPart);
        float xx = x * x;
        float v1 = c_log2f + c2 * xx;
        float v2 = x + xx * c1 * x;
        float v3 = (v2 + v1);
        *((int*)&v3) += intPart << 24;
        float v4 = v2 - v1;
        float res = v3 / (v3 - v4); 
        return res;
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
            constexpr float c1 = 0.03138777F;
            constexpr float c2 = 0.276281267F;
            constexpr float c_log2f = 1.442695022F;
            arg *= c_log2f*0.5;
            int intPart = (int)arg;
            float x = (arg - intPart);
            float xx = x * x;
            float v1 = c_log2f + c2 * xx;
            float v2 = x + xx * c1 * x;
            float v3 = (v2 + v1);
            *((int*)&v3) += intPart << 24;
            float v4 = v2 - v1;
            float res = v3 / (v3 - v4); 
            return res;
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