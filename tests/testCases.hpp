#pragma once

#include "TestCase.hpp"

#include <memory>

#include <core/core.hpp>
#include <backend/backend.hpp>
#include <test-creator/TestCreator.hpp>

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

class SimpleNeuralNetworkTest: public TestCase {
    public:
    SimpleNeuralNetworkTest() : TestCase("SimpleNeuralNetworkTest") {}
    void test(){
        using namespace neural_network;
        NeuralNetwork n({2, 5,5, 2});
        data::Data d({0, 1}, {0, 1});
        n.learn(d);
        auto firstCost = n.cost();
        n.apply();
        n.learn(d);
        auto secCost = n.cost();
        assertTrue(firstCost >= secCost);
    }
};

// normal tests
class SaveNeuralNetworkToFile: public TestCase {
    public:
    SaveNeuralNetworkToFile() : TestCase("SaveNeuralNetworkToFile") {}
    void test(){
        using namespace db;
        using namespace test_creator;
        using namespace neural_network;

        FileManager fm("net.txt");
        NeuralNetwork net({2, 20, 25, 10, 2});
        fm.to_file(net);
        std::unique_ptr<NeuralNetwork> fromFile(fm.network());
        assertEqual(net, *fromFile);

        NeuralNetwork other({2,3,2});
        assertTrue(net != other);
    }
};

class LinearBoundaryTest: public TestCase {
    public:
    LinearBoundaryTest() : TestCase("LinearBoundaryTest") {}
    void test(){
        using namespace db;
        using namespace test_creator;
        using namespace neural_network;

        NeuralNetwork net({2, 3, 4, 2});

        TestCreator creator;

        std::unique_ptr<std::vector<data::Data>> data(
            creator.prepare(
                [](double x, double y){return y > x;}
            ).createPointTest(0, 10, 512)
        );

        net.learn(data->at(0));
        auto firstCost = net.cost();
        net.learn(*data, 16);
        auto secCost = net.loss(16);

        printf(
            "\nFirst cost: %f\n" \
            "Second cost: %f\n",
            firstCost, secCost
        );

        assertTrue(firstCost > secCost);
        
    }
};


END_NAMESPACE