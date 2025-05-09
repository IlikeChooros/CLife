#include <ui/ui.hpp>
#include <optimizer/optimizer.hpp>
#include <mnist/mnist.hpp>
#include <games/games.hpp>
#include <filesystem>

static std::filesystem::path PATH;

void pointTest(){
    ui::windowWithDrawer();
}

void showTrainingDigits(){
    // testing noisy data
    mnist::Loader loader;
    auto trainingImages = loader.load(PATH / mnist::MNIST_TRAINING_SET_IMAGE_FILE_NAME).get_images();
    auto trainingLabels = loader.load(PATH / mnist::MNIST_TRAINING_SET_LABEL_FILE_NAME).get_labels();

    std::unique_ptr<data::data_batch> trainingData(
        loader.merge_data(trainingImages, trainingLabels)
    );

    // mnist::transformator t;
    // std::unique_ptr<data::data_batch> noisy(
    //     t.add_noise(trainingData.get())
    // ); 
    ui::Drawer drawer(512, 512, 14, 14, true);

    neural_network::ConvLayer layer;
    layer.initialize();
    auto pixels = layer.forward(neural_network::matrix3d_t(1, neural_network::reshape(trainingData->at(10).input, 28, 28)));    
    neural_network::MaxPoolingLayer pool;
    auto pooled = pool.forward(pixels);
    
    // pooled = layer.forward(pooled);
    // pooled = pool.forward(pooled);
    
    auto flattened = neural_network::flatten(pooled[0]);
    


    drawer.loadPixels(
        flattened
        // trainingData->at(0).input
    ).open();
}

void digitDrawerMnist(ArgParser::ParsedArguments args){
    bool train = args.get<bool>("--train");
    bool use_new = args.get<bool>("--use-new");
    std::string network_name = args.get<std::string>("--network-name");
    int epochs = args.get<int>("--epochs");
    int batch_size = args.get<int>("--batch-size");

    mnist::Loader loader;

    auto trainingImages = loader.load(PATH / mnist::MNIST_TRAINING_SET_IMAGE_FILE_NAME).get_images();
    auto trainingLabels = loader.load(PATH / mnist::MNIST_TRAINING_SET_LABEL_FILE_NAME).get_labels();
    auto testImages = loader.load(PATH / mnist::MNIST_TEST_SET_IMAGE_FILE_NAME).get_images();
    auto testLabels = loader.load(PATH / mnist::MNIST_TEST_SET_LABEL_FILE_NAME).get_labels();

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
    
    // max trainingAccuracy: 0.876 "mnistNetwork" {784, 128, 64, 10} softmax relu
    // max trainingAccuracy: ~0.89 (max: 89.9) "mnistNetwork2" {784, 255, 128, 10} softmax relu
    db::FileManager fm(network_name);

    neural_network::ONeural* network_ptr;

    constexpr int input_size = 28;

    if (use_new){
        network_ptr = new neural_network::ONeural(
            // {14*14, 256, 128, 32, 10}, 
            {input_size*input_size, 256, 128, 10}, 
            ActivationType::softmax, 
            ActivationType::relu, 
            0.2
        );
        network_ptr->initialize();
    } else {
        network_ptr = fm.from_file();
    }

    if (train)
    {
        optimizer::NeuralNetworkOptimizerParameters params;
        params.setNeuralNetwork(network_ptr)
            .setTrainingData(trainingData.get())
            .setTestData(testData.get())
            .setBatchSize(batch_size)
            .setEpochs(epochs)
            .setLearningRate(0.2);

        optimizer::NeuralNetworkOptimizer optimizer(params);
        optimizer.optimize();
    }

    std::unique_ptr<neural_network::ONeural> network(
        network_ptr
    );
    
    fm.prepare(network_name)
      .to_file(*network);

    ui::Drawer drawer(512, 512, input_size, input_size, true);

    drawer.setCallback([&](neural_network::vector_t pixels){
        network->raw_input(pixels);
        auto outputs = network->outputs();
        auto guess = network->classify();
        std::cout << "Network guess: " << guess << std::endl;
        for (size_t i = 0; i < outputs.size(); i++) {
            std::cout << '\t' << i << ": " << outputs[i] << std::endl;
        }
    }).open();
}


void cnnTest(){
    mnist::Loader loader;
    auto trainingImages = loader.load(PATH / mnist::MNIST_TRAINING_SET_IMAGE_FILE_NAME).get_images();
    auto trainingLabels = loader.load(PATH / mnist::MNIST_TRAINING_SET_LABEL_FILE_NAME).get_labels();
    auto testImages = loader.load(PATH / mnist::MNIST_TEST_SET_IMAGE_FILE_NAME).get_images();
    auto testLabels = loader.load(PATH / mnist::MNIST_TEST_SET_LABEL_FILE_NAME).get_labels();

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

    using namespace neural_network;
    cnn cnn;
    cnn.init();

    constexpr size_t epochs = 3, batchSize = 64;
    const size_t numberOfBatches = 16000 / batchSize;

    double loss[epochs] = {0.0, 0.0, 0.0};

    for (size_t i = 0; i < epochs; i++){
        for(size_t j = 0; j < numberOfBatches; j++){
            data::data_batch batch(trainingData->begin() + j * batchSize, trainingData->begin() + (j + 1) * batchSize);
            for (size_t k = 0; k < batchSize; k++){
                cnn.backprop(reshape(batch[k].input, 1, 28, 28), batch[k].expect);
                loss[i] += cnn.cost();
            }
            cnn.apply(0.4, batchSize);
        }
        std::cout << "Batch " << i << " loss: " << loss[i] / (batchSize * numberOfBatches) << std::endl;
    }

}

int main(int argc, char** argv)
{
    // cnnTest();
    PATH = std::filesystem::path(argv[0]).parent_path();

    ArgParser parser(argc, argv);

    parser.add_argument({"--network-name", "-n"}, {
        .required = false,
        .help_message = "Network name",
        .default_value = "mnist_network",
        .type = ArgParser::STRING,
        .validator = ArgParser::stringValidator,
        .action = ArgParser::store
    });

    parser.add_argument({"--train", "-t"}, {
        .required = false,
        .help_message = "Train the network",
        .default_value = "true",
        .type = ArgParser::BOOL,
        .validator = ArgParser::booleanValidator,
        .action = ArgParser::store
    });

    parser.add_argument({"--use-new", "-u"}, {
        .required = false,
        .help_message = "Use new network",
        .default_value = "true",
        .type = ArgParser::BOOL,
        .validator = ArgParser::booleanValidator,
        .action = ArgParser::store
    });

    parser.add_argument({"--batch-size", "-b"}, {
        .required = false,
        .help_message = "Batch size",
        .default_value = "64",
        .type = ArgParser::INT,
        .validator = ArgParser::intValidator,
        .action = ArgParser::store
    });

    parser.add_argument({"--epochs", "-e"}, {
        .required = false,
        .help_message = "Number of epochs",
        .default_value = "6",
        .type = ArgParser::INT,
        .validator = ArgParser::intValidator,
        .action = ArgParser::store
    });

    digitDrawerMnist(parser.parse());

    // pointTest();
    // showTrainingDigits();
    // digitDrawerMnist(true, true, "digitMT");

    // ui::Plotter plt(ui::DrawingPolicy::LineConnected);

    // plt.range(-1, 5);
    // plt.values(-1, 5);

    // for (float i = -6.3f; i < 6.3f; i += 0.2f){
    //     plt.add({i, cosf32(i)});
    // }
    
    // constexpr float T = 1.0f, N0 = 100.0f;
    // float x = -3.0f;

    // std::chrono::milliseconds dura(500);
    // auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
    // plt.addCallback([&](ui::data_t* data){
    //     auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    //     if (start + dura > now){
    //         return;
    //     }
    //     start = now;
    //     printf("Hello\n");

    //     float y = 
    //         // (expf32(-M_LN2/T * x)) * N0
    //         x + sinf32(x);
    //         ;

    //     data->push_back({x, y});
    //     x += T/4.0f;
        
    //     plt.update();
    // });
    // plt.open();

    return 0;
}