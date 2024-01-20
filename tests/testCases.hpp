#pragma once

#include "TestCase.hpp"

#include <memory>

// #include "../src/core/include.hpp"
#include <core/core.hpp>

START_NAMESPACE_TESTS

class SimpleNeuronTest: public TestCase{
    public:
    SimpleNeuronTest() : TestCase("SimpleNeuronTest") {}

    void test(){
        using namespace neural_network;
        OutputNeuron o(3);
    }
};

END_NAMESPACE