#pragma once

#include "namespaces.hpp"
#include <data/data.hpp>

#include <functional>
#include <chrono>
#include <random>

START_NAMESPACE_TEST_CREATOR


class TestCreator{
    std::function<bool(double,double)> _compFunct;

    public:
    TestCreator() = default;

    TestCreator& prepare(std::function<bool(double,double)> pointTestCompareFunction);

    std::vector<data::Data>* 
    createPointTest(double min, double max, size_t size);
};

END_NAMESPACE