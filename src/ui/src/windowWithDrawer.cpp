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
    double min, 
    double max
){
    constexpr auto PADDING = 30;
    const auto diff = float(max - min);
    const auto windowSize = std::min(window->getSize().x, window->getSize().y);
    const float nodeSize = float(windowSize - PADDING*2) / diff;

    const auto PIXEL_SIZE = nodeSize;

    auto renderingSize = windowSize / nodeSize;

    for (size_t y = 0; y < renderingSize; y++){
        for (size_t x = 0; x < renderingSize; x++){
            network->raw_input({double(x), double(y)});
            network->outputs();
            auto color = 
                network->classify() == 0
                ? sf::Color(0x02d3095F)
                : sf::Color(0xd313025F);
            auto pixel = sf::RectangleShape(sf::Vector2f(PIXEL_SIZE, PIXEL_SIZE));

            auto _x = x * PIXEL_SIZE + PADDING;
            auto _y = y * PIXEL_SIZE + PADDING;

            pixel.setPosition(sf::Vector2f(_x, _y));
            pixel.setFillColor(color);
            window->draw(pixel);
        }
    }
}

void renderPoints(
    sf::RenderWindow* window,
    std::vector<data::Data>* data,
    double min, 
    double max
){
    constexpr auto PADDING = 30;

    const auto diff = float(max - min);

    const auto windowSize = std::min(window->getSize().x, window->getSize().y);
    const float nodeSize = float(windowSize - PADDING*2) / diff;
    const float radius = nodeSize / 2;

    for (size_t i = 0; i < data->size(); i++){
        auto dataPoint = data->operator[](i);
        auto color = dataPoint.expect[0] == 1
            ? sf::Color(0x02d309FF) // green
            : sf::Color(0xd31302FF); // red

        auto x = (dataPoint.input[0] - min) * nodeSize + PADDING;
        auto y = (dataPoint.input[1] - min) * nodeSize + PADDING;

        drawNode(window, radius, x, y, color);
    }
}

void neuralNetworkPointTest(){
    using namespace sf;

    auto window = RenderWindow(VideoMode(1000, 800), "CLife");

    neural_network::ONeural net({2, 16, 16, 2}, ActivationType::relu);
    net.initialize();
    window.display();

    constexpr double MIN = 0, MAX = 100;

    test_creator::TestCreator creator;
    std::unique_ptr<std::vector<data::Data>> data(
        creator.prepare([](double x, double y){
            return (x - MAX*0.5f)*(x - MAX*0.5f) + (y - MAX*0.5f)*(y - MAX*0.5f) <= MAX*MAX*0.1f;
    }).createPointTest(MIN, MAX, 1500));

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
            // renderPoints(&window, data.get(), MIN, MAX);
            renderNetworkGuess(&window, &net, MIN, MAX);
            net.batch_learn(data.get());
            window.display();
            if(timer.getElapsedTime().asMilliseconds() >= 1000){
                timer.restart();
                printf("Cost: %f Loss: %f\n", net.cost(), net.loss());
            }
        }
    }
    return;
}

void windowWithDrawer(){
    neuralNetworkPointTest();
}

END_NAMESPACE