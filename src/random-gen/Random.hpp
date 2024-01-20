#pragma once

#include <chrono>
#include <random>
#include <stdexcept>

using urd = std::uniform_real_distribution<double>;
using dng = std::default_random_engine;

class Random
{
    dng _eng;
    urd _dist;

public:
    Random() = default;

    Random&
    prepare(double _min, double _max){
        _dist =  urd(_min, _max);

        return *this;
    }

    double
    rand(){
        _eng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        return _dist(_eng);
    }
};