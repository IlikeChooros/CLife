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

    const Data& operator=(const Data& other){
        this->input = other.input;
        this->expect = other.expect;
        return *this;
    }
};

class Point{

public:
    Point()=default;

    Point(int x, int y):
    x(x), y(y) {}

    int x;
    int y;
};

typedef std::vector<Data> data_batch;
typedef std::vector<std::vector<double>> matrix_t;

END_NAMESPACE