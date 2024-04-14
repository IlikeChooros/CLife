#pragma once

#include "namespaces.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

#include <core/core.hpp>
#include <mnist/mnist.hpp>
#include <ui/Plotter.hpp>

START_NAMESPACE_OPTIMIZER

struct NeuralNetworkOptimizerParameters
{
    NeuralNetworkOptimizerParameters(): 
        batchSize(0), epochs(0), learningRate(0.4),
        trainingData(nullptr), testData(nullptr), network(nullptr) {};
    virtual ~NeuralNetworkOptimizerParameters() {};

    NeuralNetworkOptimizerParameters& setNeuralNetwork(
        neural_network::ONeural* network
    ) {this->network = network; return *this;}

    NeuralNetworkOptimizerParameters& setTrainingData(
        data::data_batch* trainingData
    ) {this->trainingData = trainingData; return *this;}

    NeuralNetworkOptimizerParameters& setTestData(
        data::data_batch* testData
    ) {this->testData = testData; return *this;}

    NeuralNetworkOptimizerParameters& setBatchSize(
        size_t batchSize
    ) {this->batchSize = batchSize; return *this;}

    NeuralNetworkOptimizerParameters& setEpochs(
        size_t epochs
    ) {this->epochs = epochs; return *this;}

    NeuralNetworkOptimizerParameters& setLearningRate(
        double learningRate
    ) {this->learningRate = learningRate; return *this;}

    size_t batchSize;
    size_t epochs;
    double learningRate;
    data::data_batch* trainingData;
    data::data_batch* testData;
    neural_network::ONeural* network;
};

struct NeuralNetworkOptimizerResult
{
    NeuralNetworkOptimizerResult() = default;
    virtual ~NeuralNetworkOptimizerResult() = default;

    NeuralNetworkOptimizerResult& setTrainingAccuracy(
        double trainingAccuracy
    ) {this->trainingAccuracy = trainingAccuracy; return *this;}

    NeuralNetworkOptimizerResult& setTestAccuracy(
        double testAccuracy
    ) {this->testAccuracy = testAccuracy; return *this;}

    double trainingAccuracy;
    double testAccuracy;
};

class NeuralNetworkOptimizer
{
public:
    NeuralNetworkOptimizer() = default;
    NeuralNetworkOptimizer(const NeuralNetworkOptimizerParameters& params);
    virtual ~NeuralNetworkOptimizer() = default;

    NeuralNetworkOptimizer& setParameters(
        const NeuralNetworkOptimizerParameters& params
    );

    NeuralNetworkOptimizerResult optimize();
    NeuralNetworkOptimizerResult partialOptimize(
        size_t epochs = 0UL, size_t batchSize = 32UL
    );

    double train_epoch(size_t total_batches, ui::Plotter* = nullptr);

private:
    NeuralNetworkOptimizerParameters params;
};

END_NAMESPACE_OPTIMIZER