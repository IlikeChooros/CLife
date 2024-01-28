#include <optimizer/optimizer.hpp>


START_NAMESPACE_OPTIMIZER

NeuralNetworkOptimizer::NeuralNetworkOptimizer(
    const NeuralNetworkOptimizerParameters& params)
{
    this->params = params;
}

NeuralNetworkOptimizer& NeuralNetworkOptimizer::setParameters(
    const NeuralNetworkOptimizerParameters& params)
{
    this->params = params;
    return *this;
}

double NeuralNetworkOptimizer::train_epoch(size_t total_batches)
{
    double average_loss = 0.0;
    for (size_t i = 0; i < total_batches; ++i)
    {
        params.network->batch_learn(
            params.trainingData, 
            params.batchSize, 
            params.learningRate
        );
        average_loss += params.network->loss(
            params.batchSize
        );
    }
    return average_loss / total_batches;
}

NeuralNetworkOptimizerResult NeuralNetworkOptimizer::optimize()
{
    NeuralNetworkOptimizerResult result;
    result.setTrainingAccuracy(0.0);
    result.setTestAccuracy(0.0);

    size_t total_batches = params.trainingData->size() / params.batchSize;

    for (size_t i = 0; i < params.epochs; ++i)
    {
        auto average_loss = train_epoch(total_batches);
        result.setTrainingAccuracy(average_loss);
    }

    result.setTestAccuracy(params.network->accuracy(
        params.testData
    ));

    return result;
}

END_NAMESPACE_OPTIMIZER