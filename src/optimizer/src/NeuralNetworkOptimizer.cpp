#include <optimizer/optimizer.hpp>

START_NAMESPACE_OPTIMIZER

NeuralNetworkOptimizer::NeuralNetworkOptimizer(
    const NeuralNetworkOptimizerParameters &params)
{
    this->params = params;
}

NeuralNetworkOptimizer &NeuralNetworkOptimizer::setParameters(
    const NeuralNetworkOptimizerParameters &params)
{
    this->params = params;
    return *this;
}

double NeuralNetworkOptimizer::train_epoch(size_t total_batches, ui::Visualizer &visualizer, size_t start_time)
{
    std::shuffle(
        params.trainingData->begin(),
        params.trainingData->end(),
        std::mt19937(std::random_device()()));

    double average_loss = 0.0;
    double current_loss = 0.0;
    int64_t total_time = start_time;

    for (size_t i = 0; i < total_batches; ++i)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        params.network->batch_learn(
            params.trainingData,
            params.learningRate,
            params.batchSize);
        current_loss = params.network->loss(
            params.batchSize);
        average_loss += current_loss;

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - startTime)
                         .count();
        total_time += delta;

        visualizer.update({(int)i, current_loss, total_time});
        visualizer.visualize();
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
    size_t totalTime = 0, timeDiff = 0;

    ui::GraphVisualizer visualizer;

    for (size_t i = 0; i < params.epochs; ++i)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        auto average_loss = train_epoch(total_batches, visualizer, totalTime);
        result.setTrainingAccuracy(1 - average_loss);

        timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now() - startTime)
                       .count();
        totalTime += timeDiff;
        std::cout << "**** Epoch " << i << " average loss: " << average_loss << " time: "
                  << timeDiff << "ms" << std::endl;
    }

    result.setTestAccuracy(params.network->accuracy(params.testData));
    auto recallNormal = params.network->recall(params.testData);

    mnist::transformator t;
    auto noisy_acc = params.network->accuracy(t.add_noise(params.testData, 4, 28, 28, 100, false));
    auto recallNoisy = params.network->recall(params.testData);
    
    std::cout << "Training complete. Learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count()
              << "ms \n"
              << "Accuracy original: " << result.testAccuracy << "\n"
                << "Accuracy noisy: " << noisy_acc << "\n"
                << "Recall original: \n";
    
    auto printRecall = [](const data::vector_t &recall) {
        for (size_t i = 0; i < recall.size(); i++)
        {
            std::cout << "\t" << i << ": " << recall[i] << "\n";
        }
    };

    printRecall(recallNormal);
    std::cout << "Recall noisy: \n";
    printRecall(recallNoisy);


    return result;
}

END_NAMESPACE_OPTIMIZER