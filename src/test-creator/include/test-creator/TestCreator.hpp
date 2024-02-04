#pragma once

#include "namespaces.hpp"
#include <data/data.hpp>

#include <functional>
#include <chrono>
#include <random>
#include <memory>

START_NAMESPACE_TEST_CREATOR


class TestCreator{
    std::function<bool(double,double)> _compFunct;

    public:
    TestCreator() = default;

    TestCreator& prepare(std::function<bool(double,double)> pointTestCompareFunction);

    data::data_batch* 
    createPointTest(double min, double max, size_t size);
};

END_NAMESPACE