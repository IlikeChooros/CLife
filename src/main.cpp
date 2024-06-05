#include <ui/ui.hpp>
#include <optimizer/optimizer.hpp>
#include <mnist/mnist.hpp>
#include <games/games.hpp>

#include <filesystem>

void pointTest()
{
    ui::windowWithDrawer();
}

void showTrainingDigits()
{
    // testing noisy data
    const std::string PATH =
        // "/home/minis/Desktop/CLife/src/mnist/digits/";
        "C:\\Users\\pawel\\OneDrive\\Desktop\\CLife\\src\\mnist\\digits\\";
    mnist::Loader loader;
    auto trainingImages = loader.load(PATH + mnist::MNIST_TRAINING_SET_IMAGE_FILE_NAME).get_images();
    auto trainingLabels = loader.load(PATH + mnist::MNIST_TRAINING_SET_LABEL_FILE_NAME).get_labels();

    std::unique_ptr<data::data_batch> trainingData(
        loader.merge_data(trainingImages, trainingLabels));

    std::cout << "MNIST dataset loaded\n" 
              << "Use original dataset? [y/n] >> ";
    std::string use_original;
    std::cin >> use_original;

    while(use_original != "y" && use_original != "n"){
        std::cout << "Invalid option\n";
        std::cout << "Use original dataset? [y/n] >> ";
        std::cin >> use_original;
    }

    if (use_original == "n"){
        mnist::transformator t;
        t.add_noise(trainingData.get(), 4, 28, 28, 100, false);
    }

    ui::Drawer drawer;
    drawer.showImages(*trainingData);

    system("cls");
}

void digitDrawerMnist(bool use_new = false, bool train = false, std::string network_name = "original")
{
    const std::string PATH =
        //  "/home/minis/Desktop/CLife/src/mnist/digits/";
        "C:\\Users\\pawel\\OneDrive\\Desktop\\CLife\\src\\mnist\\digits\\";
    mnist::Loader loader;
    auto trainingImages = loader.load(PATH + mnist::MNIST_TRAINING_SET_IMAGE_FILE_NAME).get_images();
    auto trainingLabels = loader.load(PATH + mnist::MNIST_TRAINING_SET_LABEL_FILE_NAME).get_labels();
    auto testImages = loader.load(PATH + mnist::MNIST_TEST_SET_IMAGE_FILE_NAME).get_images();
    auto testLabels = loader.load(PATH + mnist::MNIST_TEST_SET_LABEL_FILE_NAME).get_labels();

    std::cout << trainingImages->size() << std::endl;
    std::cout << trainingLabels->size() << std::endl;
    std::cout << testImages->size() << std::endl;
    std::cout << testLabels->size() << std::endl;


    std::cout << "MNIST dataset loaded\n" 
            << "Use original dataset? [y/n] >> ";
    std::string use_original = "y";
    std::cin >> use_original;

    while(use_original != "y" && use_original != "n"){
        std::cout << "Invalid option\n";
        std::cout << "Use original dataset? [y/n] >> ";
        std::cin >> use_original;
    }

    data::data_batch* training;
    if (use_original == "n"){
        mnist::transformator t;
        training = t.add_noise(loader.merge_data(trainingImages, trainingLabels), 4, 28, 28, 100, false);
    } else {
        training = loader.merge_data(trainingImages, trainingLabels);
    }


    std::string use_new_str = "n", train_str = "y";
    std::cout << "Use new network? [y/n] >> ";
    std::cin >> use_new_str;

    while(use_new_str != "y" && use_new_str != "n"){
        std::cout << "Invalid option\n";
        std::cout << "Use new network? [y/n] >> ";
        std::cin >> use_new_str;
    }

    use_new = use_new_str == "y";

    
    std::cout << "Network name? >> ";
    std::cin >> network_name;
    

    std::cout << "Train network? [y/n] >> ";
    std::cin >> train_str;

    while(train_str != "y" && train_str != "n"){
        std::cout << "Invalid option\n";
        std::cout << "Train network? [y/n] >> ";
        std::cin >> train_str;
    }

    train = train_str == "y";

    std::unique_ptr<data::data_batch> trainingData(training);
    std::unique_ptr<data::data_batch> testData(
        loader.merge_data(testImages, testLabels));

    std::cout << trainingData->size() << std::endl;
    std::cout << testData->size() << std::endl;

    std::cout << trainingData->at(0).input.size() << std::endl;
    std::cout << trainingData->at(0).expect.size() << std::endl;

    // max trainingAccuracy: 0.876 "mnistNetwork" {784, 128, 64, 10} softmax relu
    // max trainingAccuracy: ~0.89 (max: 89.9) "mnistNetwork2" {784, 255, 128, 10} softmax relu

    db::FileManager fm(network_name);

    neural_network::ONeural *network_ptr;

    if (use_new){
        network_ptr = new neural_network::ONeural({784, 256, 128, 32, 10}, ActivationType::softmax, ActivationType::relu);
        network_ptr->initialize();
    }
    else{
        network_ptr = fm.from_file();
    }

    if (train){

        auto printOptions = [](){
            std::cout << "Training options: \n"
                  << "(f) Fast\n"
                  << "(d) Default\n"
                  << "(l) Long \n"
                  << "(c) Custom \n"
                  << "Option (f,d,l,c) >> ";
        };
        printOptions();
        std::string tr_opt;
        std::cin >> tr_opt;

        while(tr_opt != "f" && tr_opt != "d" && tr_opt != "l" && tr_opt != "c"){
            if (tr_opt == ""){
                tr_opt = "d";
                break;
            }
            std::cout << "Invalid option\n";
            printOptions();
            std::cin >> tr_opt;
        }

        optimizer::NeuralNetworkOptimizerParameters params;
        params.setNeuralNetwork(network_ptr)
            .setTrainingData(trainingData.get())
            .setTestData(testData.get())
            .setBatchSize(64)
            .setEpochs(1)
            .setLearningRate(0.1);

        if (tr_opt == "f"){
            params.setBatchSize(64)
                .setEpochs(1)
                .setLearningRate(0.3);
        } else if (tr_opt == "d"){
            params.setBatchSize(64)
                .setEpochs(2)
                .setLearningRate(0.25);
        } else if (tr_opt == "l"){
            params.setBatchSize(64)
                .setEpochs(5)
                .setLearningRate(0.1);
        } else{
            std::cout << "Batch size >> ";
            int batch_size;
            std::cin >> batch_size;
            std::cout << "Epochs >> ";
            int epochs;
            std::cin >> epochs;
            std::cout << "Learning rate >> ";
            float learning_rate;
            std::cin >> learning_rate;

            params.setBatchSize(batch_size)
                .setEpochs(epochs)
                .setLearningRate(learning_rate);
        }

        optimizer::NeuralNetworkOptimizer optimizer(params);
        optimizer.optimize();
    }

    std::unique_ptr<neural_network::ONeural> network(
        network_ptr);

    fm.prepare(network_name)
        .to_file(*network);

    ui::Drawer drawer;

    drawer.setCallback([&](neural_network::vector_t pixels)
                       {
        network->raw_input(pixels);
        auto outputs = network->outputs();
        auto guess = network->classify();
        std::cout << "Network guess: " << guess << std::endl;
        for (size_t i = 0; i < outputs.size(); i++) {
            std::cout << '\t' << i << ": " << outputs[i] << std::endl;
        } })
        .open();
}

void printOptions()
{
    std::cout << "Pick option: \n"
              << "1. Point test\n"
              << "2. Show training digits\n"
              << "3. Digit recognition\n"
              << "4. Exit\n";
    std::cout << "Option [1-4] >> ";
}

int main()
{
    printOptions();

    std::string option = "3";
    std::cin >> option;

    while (option != "4" && option != "exit" && option != "q" && option != "quit")
    {
        system("cls");
        if (option == "1")
        {
            pointTest();
        }
        else if (option == "2")
        {
            showTrainingDigits();
        }
        else if (option == "3")
        {
            digitDrawerMnist();
        }
        else
        {
            std::cout << "Invalid option\n";
        }
        printOptions();
        std::cin >> option;
    }

    std::cout << "Exiting...\n";

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