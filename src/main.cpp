#include <ui/ui.hpp>
#include <optimizer/optimizer.hpp>
#include <mnist/mnist.hpp>

int main()
{
    const std::string PATH = "/home/minis/Desktop/CLife/src/mnist/digits/";
    mnist::Loader loader;
    auto trainingImages = loader.load(PATH + mnist::MNIST_TRAINING_SET_IMAGE_FILE_NAME).get_images();
    auto trainingLabels = loader.load(PATH + mnist::MNIST_TRAINING_SET_LABEL_FILE_NAME).get_labels();
    auto testImages = loader.load(PATH + mnist::MNIST_TEST_SET_IMAGE_FILE_NAME).get_images();
    auto testLabels = loader.load(PATH + mnist::MNIST_TEST_SET_LABEL_FILE_NAME).get_labels();

    std::cout << trainingImages->size() << std::endl;
    std::cout << trainingLabels->size() << std::endl;
    std::cout << testImages->size() << std::endl;
    std::cout << testLabels->size() << std::endl;

    std::unique_ptr<data::data_batch> trainingData(
        loader.merge_data(trainingImages, trainingLabels)
    );
    std::unique_ptr<data::data_batch> testData(
        loader.merge_data(testImages, testLabels)
    );

    // delete trainingImages;
    // delete trainingLabels;
    // delete testImages;
    // delete testLabels;

    std::cout << trainingData->size() << std::endl;
    std::cout << testData->size() << std::endl;

    std::cout << trainingData->at(0).input.size() << std::endl;
    std::cout << trainingData->at(0).expect.size() << std::endl;

    // // testing noisy data
    // mnist::transformator t;
    // std::unique_ptr<data::data_batch> noisy(t.add_noise(trainingData.get()));
    // ui::Drawer drawer;
    // drawer.loadPixels(noisy->at(0).input).open();
    

    db::FileManager fm("mnist_network6-saved.txt");

    std::unique_ptr<neural_network::ONeural> network(
        fm.from_file()
        // new neural_network::ONeural({784, 128, 64, 10})
    );  
    network->initialize();

    optimizer::NeuralNetworkOptimizerParameters params;
    params.setNeuralNetwork(network.get())
          .setTrainingData(trainingData.get())
          .setTestData(testData.get())
          .setBatchSize(64)
          .setEpochs(4)
          .setLearningRate(0.72);

    optimizer::NeuralNetworkOptimizer optimizer(params);
    optimizer.optimize();
    
    fm.prepare("mnist_network6-saved.txt")
      .to_file(*network);

    ui::Drawer drawer;

    size_t prevGuess = 0;
    drawer.setCallback([&](std::vector<double> pixels){
        network->raw_input(pixels);
        auto outputs = network->outputs();
        auto guess = network->classify();
        // if (guess != prevGuess) {
            std::cout << "Network guess: " << guess << std::endl;
            for (size_t i = 0; i < outputs.size(); i++) {
                std::cout << '\t' << i << ": " << outputs[i] << std::endl;
            }
            prevGuess = guess;
        // }
    }).open();

    return 0;
}