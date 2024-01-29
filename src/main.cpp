#include <ui/ui.hpp>
#include <optimizer/optimizer.hpp>
#include <mnist/mnist.hpp>

int main()
{
    const std::string PATH = "src/mnist/digits/";
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


    optimizer::NeuralNetworkOptimizerParameters params;
    test_creator::TestCreator creator;

    std::unique_ptr<neural_network::ONeural> network(
        new neural_network::ONeural({784, 128, 64, 10})
    );  
    network->initialize();

    params.setNeuralNetwork(network.get()).setTrainingData(trainingData.get())
          .setTestData(testData.get()).setBatchSize(128).setEpochs(2).setLearningRate(0.15);

    optimizer::NeuralNetworkOptimizer optimizer(params);
    optimizer.train_epoch(5);
    
    db::FileManager fm("mnist_network.txt");
    fm.to_file(*network);




    // optimizer::NeuralNetworkOptimizer optimizer;

    // auto testCmp = [](double x, double y) -> bool {
    //     return x > y;
    // };

    // optimizer::NeuralNetworkOptimizerParameters params;
    // test_creator::TestCreator creator;
    // std::unique_ptr<data::data_batch> trainingData(
    //     creator.prepare(testCmp).createPointTest(0, 200, 10000)
    // );
    // std::unique_ptr<data::data_batch> testData(
    //     creator.createPointTest(0, 200, 2048)
    // );

    // std::unique_ptr<neural_network::ONeural> network(
    //     new neural_network::ONeural({2, 16, 16, 2})
    // );
    // network->initialize();

    // params.setNeuralNetwork(network.get()).setTrainingData(trainingData.get())
    //       .setTestData(testData.get()).setBatchSize(32).setEpochs(2).setLearningRate(0.15);
        
    // optimizer.setParameters(params);
    // optimizer.optimize();


    // ui::windowWithDrawer();
    return 0;
}