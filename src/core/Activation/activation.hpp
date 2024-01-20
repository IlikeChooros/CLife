#pragma once

#include <algorithm>



class BaseActivation{
public:
    virtual double activation(double arg) = 0;
    virtual double derivative(double activation) = 0;
};

class Sigmoid: public BaseActivation{
public:
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

class ReLu: public BaseActivation{
public:
    double activation(double arg){
        return std::max((double)0, arg);
    }
    double derivative(double activation){
        return activation > 0 ? 1 : 0;
    }
};