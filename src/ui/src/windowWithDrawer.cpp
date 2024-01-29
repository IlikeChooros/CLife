#include <ui/windowWithDrawer.hpp>

START_NAMESPACE_UI

void baseWindowPlayer(
    std::function<void(sf::RenderWindow&, neural_network::NeuralNetwork&)> func
){
    using namespace sf;

    auto window = RenderWindow(VideoMode(1000, 800), "CLife");

    neural_network::NeuralNetwork net({2, 5, 3, 5, 5, 7, 2, 3, 2});
        
    window.setFramerateLimit(60);
    window.display();

    while(window.isOpen()){
        Event event; 
        if(window.pollEvent(event)){
            switch (event.type)
            {
            case Event::Closed:
                window.close();
                break;
            
            default:
                break;
            }
        }
        func(window, net);
    }
    return;
}



void drawNode(sf::RenderWindow* window, float radius, float posX, float posY, const sf::Color& color = sf::Color(250, 100 ,100)){
    sf::CircleShape node(radius);

    node.setPosition({posX, posY});
    node.setFillColor(color);

    window->draw(node);
}

float getStartY(float height, float maxElements, float n){
    return height * (1 - n / maxElements) / 2;
}

void drawLayer(
    sf::RenderWindow* window, neural_network::BaseLayer* layer, int layerIndex,
    int maxElements, float networkHeight,
    float radius, float startX, float startY, float SPACE
){

    startY = startY + getStartY(networkHeight, (float)maxElements, (float)layer->node_out);
    for (int n = 0; n < layer->node_out; n++){
        drawNode(
            window, radius, 
            startX + layerIndex * (radius* 2 + SPACE),
            startY + n*(radius* 2 + SPACE) 
        ); 
    }
}

void neuralNetworkVisualization(
    sf::RenderWindow& window, neural_network::NeuralNetwork& network
){
    window.clear();

    constexpr auto PADDING = 200, SPACE = 30;

    auto dimensions = window.getSize();
    auto strucutre = network.raw_structure();


    auto maxElements = *std::max_element(strucutre.begin(), strucutre.end());
    float networkWidth = (dimensions.x - 2*PADDING) / strucutre.size() - SPACE;
    float networkHeight = (dimensions.y - 2*PADDING) / maxElements - SPACE;

    const float radius = std::min(networkWidth, networkHeight) / 2;
    const float startX = PADDING;
    const float startY = PADDING;
    const float renderingWindowHeight = dimensions.y - 2*PADDING;


    // neural network visualization

    // inputs
    const float centeredY = startY + getStartY(renderingWindowHeight, maxElements, strucutre[0]);
    for (int i = 0; i < strucutre[0]; i++){
        drawNode(&window, radius, startX, centeredY + i * (2*radius + SPACE));
    }

    // hidden layer
    for (int layer = 0; layer < network._hidden_size; layer++){
        drawLayer(
            &window, network._hidden_layer[layer], layer + 1,
            maxElements, renderingWindowHeight,
            radius, startX, startY, SPACE
        );
    }

    // output layer
    drawLayer(
        &window, network._output_layer, 
        network._hidden_size + 1,
        maxElements, renderingWindowHeight,
        radius, startX, startY, SPACE
    );

    window.display();
}

void renderNetworkGuess(
    sf::RenderWindow* window, 
    neural_network::ONeural* network,
    neural_network::data_batch* data,
    double min, 
    double max
){
   constexpr auto radius = 16.0f;

    sf::Vector2u windowSize = window->getSize();

    auto minSize = std::min(windowSize.x, windowSize.y);

    for (size_t i = 0; i < data->size(); i++){
        auto dataPoint = (*data)[i];
        network->input(dataPoint);
        network->outputs();
        auto color = network->correct()
            ? sf::Color(0x02d30940) // green
            : sf::Color(0xd3130240); // red

        auto x = ((dataPoint.input[0] - min) / (max - min)) * minSize - radius/2;
        auto y = ((dataPoint.input[1] - min) / (max - min)) * minSize - radius/2;

        drawNode(window, radius, x, y, color);
    }
}

void renderPoints(
    sf::RenderWindow* window,
    std::vector<data::Data>* data,
    neural_network::ONeural* network,
    double min, 
    double max
){
    constexpr auto PADDING = 20;
    constexpr auto radius = 5.0f;

    sf::Vector2u windowSize = window->getSize();

    auto minSize = std::min(windowSize.x, windowSize.y);
    minSize -= 2*PADDING;

    for (size_t i = 0; i < data->size(); i++){
        auto dataPoint = (*data)[i];
        auto color = dataPoint.expect[0] == 1
            ? sf::Color(0x02d309FF) // green
            : sf::Color(0xd31302FF); // red

        auto x = dataPoint.input[0] * minSize + PADDING;
        auto y = (1 - dataPoint.input[1]) * minSize + PADDING;

        drawNode(window, radius, x, y, color);

        network->input(dataPoint);
        network->outputs();
        auto netColor = network->classify() == 0
            ? sf::Color(0x02d30940) // green
            : sf::Color(0xd3130240); // red
        
        drawNode(window, 2.5f*radius, x - radius*0.5f, y - radius*0.5f, netColor);
    }
}

void neuralNetworkPointTest(){
    using namespace sf;

    auto window = RenderWindow(VideoMode(1000, 800), "CLife");

    neural_network::ONeural net = neural_network::ONeural(
        {2,16,8,4,2}, ActivationType::softmax, ActivationType::relu);
    net.initialize();
    window.display();

    constexpr double MIN = 0, MAX = 400;

    test_creator::TestCreator creator;
    std::unique_ptr<std::vector<data::Data>> data(
        creator.prepare([](double x, double y){
            return (MAX*0.000025)*x*x - (MAX*0.01)*x + (MAX*1.2)> y;
            // return y < 100 || y > 300;
            // return (x - MAX*0.5f)*(x - MAX*0.5f) + (y - MAX*0.5f)*(y - MAX*0.5f) <= MAX*MAX*0.1f;
            // return 0.0018*x*x - 6*x + MAX*1.2 > y;
            // return (MAX*0.000025)*x*x - (MAX*0.01)*x + (MAX*1.2)> y;
    }).createPointTest(MIN, MAX, 1024));

    Clock timer;

    Clock displayTimer;
    while(window.isOpen()){
        Event event; 
        if(window.pollEvent(event)){
            switch (event.type)
            {
            case Event::Closed:{
                window.close();
                db::FileManager fm("netw.txt");
                fm.to_file(net);
            }
                break;
            
            default:
                break;
            }
        }  
        auto time = displayTimer.getElapsedTime().asMilliseconds();
        if(time >= 20){
            displayTimer.restart();
            window.clear();
            renderPoints(&window, data.get(), &net, MIN, MAX);
            // renderNetworkGuess(&window, &net, data.get(), MIN, MAX);
            
            // net.raw_input({double(Mouse::getPosition(window).x), double(Mouse::getPosition(window).y)});
            window.display();
            if(timer.getElapsedTime().asMilliseconds() >= 1000){
                timer.restart();
                auto dataPoint = (*data)[rand() % data->size()];
                net.input(dataPoint);
                auto outputs = net.outputs();
                printf(
                    "Cost: %f Loss: %f Output0: %f Output1: %f classify: %lu x: %f y: %f\n",
                     net.cost(), net.loss(), outputs[0], outputs[1], net.classify(),
                    dataPoint.input[0] * (MAX- MIN), dataPoint.input[1] * (MAX - MIN)
                );
            }
        }
        net.batch_learn(data.get(), 0.6, 64);
        
    }
    return;
}

void windowWithDrawer(){
    srand(time(NULL));
    neuralNetworkPointTest();
}

END_NAMESPACE