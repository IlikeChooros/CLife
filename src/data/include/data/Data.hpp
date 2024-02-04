#pragma once

#include "types.hpp"


START_NAMESPACE_DATA



/// @brief Input class for neural network
class Data{
    public:
    Data() = default;
    
    Data(const vector_t& inputs, const vector_t& exp){
        input = inputs;
        expect = exp;
    }

    vector_t input;
    vector_t expect;

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

END_NAMESPACE