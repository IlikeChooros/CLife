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
    double current_loss = 0.0;
    for (size_t i = 0; i < total_batches; ++i)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        params.network->batch_learn(
            params.trainingData, 
            params.learningRate,
            params.batchSize
        );
        current_loss = params.network->loss(
            params.batchSize
        );
        average_loss += current_loss;

        std::cout << "Batch " << i << " loss: " << current_loss << " time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count() 
            << "us" << std::endl;
    }
    return average_loss / total_batches;
}

NeuralNetworkOptimizerResult NeuralNetworkOptimizer::optimize()
{
    NeuralNetworkOptimizerResult result;
    result.setTrainingAccuracy(0.0);
    result.setTestAccuracy(0.0);

    auto startTime = std::chrono::high_resolution_clock::now();
    std::cout << "Training network..." << std::endl;

    size_t total_batches = params.trainingData->size() / params.batchSize;

    for (size_t i = 0; i < params.epochs; ++i)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        auto average_loss = train_epoch(total_batches);
        result.setTrainingAccuracy(average_loss);

        std::cout << "**** Epoch " << i << " average loss: " << average_loss << " time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count() 
            << "ms" << std::endl;
    }

    result.setTestAccuracy(params.network->accuracy(
        params.testData
    ));

    std::cout << "Training complete. Learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startTime).count() 
        << "ms " << "trainingAccuracy: " << result.trainingAccuracy 
        << " testAccuracy: " << result.testAccuracy << std::endl;

    return result;
}

END_NAMESPACE_OPTIMIZER