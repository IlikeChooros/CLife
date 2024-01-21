#pragma once

#include <vector>

#include "namespaces.hpp"


START_NAMESPACE_DATA
    

/// @brief Input class for neural network
class Data{
    public:
    Data() = default;
    
    Data(const std::vector<double>& inputs, const std::vector<double>& exp){
        input = inputs;
        expect = exp;
    }

    std::vector<double> input;
    std::vector<double> expect;
};

class Point{

public:
    Point()=default;

    Point(int x, int y):
    x(x), y(y) {}

    int x;
    int y;
};

END_NAMESPACE