#pragma once

#include "TestCase.hpp"

#include <memory>

#include <core/core.hpp>

START_NAMESPACE_TESTS

class SimpleNeuronTest: public TestCase{
    public:
    SimpleNeuronTest() : TestCase("SimpleNeuronTest") {}

    void test(){
        using namespace neural_network;
        OutputNeuron o(3);
        o.set_inputs({1,-0.7, 0.4});
        auto firstAct = o.activation();
        o.calculate_gradient(1);
        o.apply_gradients();
        assertTrue(o.activation() >= firstAct);
    }
};

class SimpleLayerTest: public TestCase{
    public:
    SimpleLayerTest() : TestCase("SimpleLayerTest") {}
    void test(){
        using namespace neural_network;
        OutputLayer l(5, 2);
        l.set_inputs({1,-0.7, 0.4, 0.5, -0.2});
        auto firstAct = l.calc_outputs();
        l.calc_gradient({0,1});
        l.apply_gradient(1, 1);
        auto secAct = l.calc_outputs();

        assertTrue(firstAct[0] > secAct[0]); // should be smaller
        assertTrue(firstAct[1] < secAct[1]); // should be bigger
    }
};

END_NAMESPACE