#include <ui/ui.hpp>
#include <optimizer/optimizer.hpp>
#include <mnist/mnist.hpp>

int main()
{

    // data::DoodlesLoader loader;
    // std::unique_ptr<data::matrix_t> image (loader.load(
    //     "/home/minis/Desktop/CLife/src/data/doodles/full_binary_airplane.bin").get_images());

    // ui::Drawer drawer(512*2, 512*2, 256, 256);
    // drawer.loadPixels(image->at(0)).open();



    // db::FileManager fm("binary-test");
    // neural_network::ONeural ne({2, 64, 10});
    // ne.initialize();
    // fm.prepare("binary-test").to_file(ne);

    // std::unique_ptr<neural_network::ONeural> ne2(fm.from_file());



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

    delete trainingImages;
    delete trainingLabels;
    delete testImages;
    delete testLabels;

    std::cout << trainingData->size() << std::endl;
    std::cout << testData->size() << std::endl;

    std::cout << trainingData->at(0).input.size() << std::endl;
    std::cout << trainingData->at(0).expect.size() << std::endl;

    // // testing noisy data
    // mnist::transformator t;
    // std::unique_ptr<data::data_batch> noisy(t.add_noise(trainingData.get()));
    // ui::Drawer drawer;
    // drawer.loadPixels(noisy->at(4).input).open();
    

    db::FileManager fm("mnistNetwork");

    std::unique_ptr<neural_network::ONeural> network(
        fm.from_file()
        // new neural_network::ONeural({784, 128, 64, 10}
        //     , ActivationType::softmax, ActivationType::relu
        // // , ActivationType::softmax, ActivationType::sigmoid
        // )
    );  
    // network->initialize();

    optimizer::NeuralNetworkOptimizerParameters params;
    params.setNeuralNetwork(network.get())
          .setTrainingData(trainingData.get())
          .setTestData(testData.get())
          .setBatchSize(128)
          .setEpochs(5)
          .setLearningRate(0.3);

    optimizer::NeuralNetworkOptimizer optimizer(params);
    optimizer.optimize();
    
    fm.prepare("mnistNetwork")
      .to_file(*network);

    ui::Drawer drawer;

    drawer.setCallback([&](std::vector<double> pixels){
        network->raw_input(pixels);
        auto outputs = network->outputs();
        auto guess = network->classify();
        std::cout << "Network guess: " << guess << std::endl;
        for (size_t i = 0; i < outputs.size(); i++) {
            std::cout << '\t' << i << ": " << outputs[i] << std::endl;
        }
    }).open();

    return 0;
}