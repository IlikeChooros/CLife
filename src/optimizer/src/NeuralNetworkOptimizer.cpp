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

std::vector<ui::_PlotPoint> losses;
std::mutex loss_mutex;


void plot_loss(ui::Plotter* plotter)
{
    plotter->addCallback([&](ui::data_t* data){
        std::lock_guard<std::mutex> lock(loss_mutex);
        *data = losses;
        plotter->update();
    });
    plotter->open();
}

double NeuralNetworkOptimizer::train_epoch(size_t total_batches, ui::Plotter* plotter)
{
    std::shuffle(
        params.trainingData->begin(), 
        params.trainingData->end(), 
        std::mt19937(std::random_device()())
    );
    mnist::transformator t;
    std::unique_ptr<data::data_batch> noisy(
        t.add_noise(params.trainingData, 5)
    );
    double average_loss = 0.0;
    double current_loss = 0.0;
    int64_t total_time = 0;

    for (size_t i = 0; i < total_batches; ++i)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        params.network->batch_learn(
            noisy.get(), 
            params.learningRate,
            params.batchSize
        );
        current_loss = params.network->loss(
            params.batchSize
        );
        average_loss += current_loss;

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - startTime
        ).count();
        total_time += delta;

        std::cout << "Batch " << i << " loss: " << current_loss << " time: " 
            << delta
            << "ms" << std::endl;
        
        std::lock_guard<std::mutex> lock(loss_mutex);
        losses.push_back({(float)total_time, (float)current_loss});
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

    // ui::Plotter plotter(ui::DrawingPolicy::LineConnected);
    // plotter.values(0, 1.1);

    losses.clear();
    losses.reserve(total_batches * params.epochs);

    // std::thread plot_thread(plot_loss, &plotter);

    for (size_t i = 0; i < params.epochs; ++i)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        auto average_loss = train_epoch(total_batches);
        result.setTrainingAccuracy(1 - average_loss);

        std::cout << "**** Epoch " << i << " average loss: " << average_loss << " time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count() 
            << "ms" << std::endl;
    }

    result.setTestAccuracy(params.network->accuracy(
        params.testData
    ));

    mnist::transformator t;
    std::unique_ptr<data::data_batch> noisy(
        t.add_noise(params.testData, 5)
    );
    auto noisy_acc = params.network->accuracy(noisy.get());

    std::cout << "Training complete. Learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startTime).count() 
        << "ms " << "trainingAccuracy: " << result.trainingAccuracy << std::endl
        << "\tNoisy Test Accuracy: " << noisy_acc << std::endl
        << "\tTest Acc: "<< result.testAccuracy << std::endl;

    return result;
}

END_NAMESPACE_OPTIMIZER