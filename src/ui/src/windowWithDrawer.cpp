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

void neuralNetworkPointTestVisalization(
    sf::RenderWindow* window, neural_network::NeuralNetwork* network,
    std::vector<data::Data>* data, double min, double max, size_t time
){
    window->clear();

    constexpr auto PADDING = 30;

    const auto diff = float(max - min);

    const auto windowSize = std::min(window->getSize().x, window->getSize().y);
    const float nodeSize = float(windowSize - PADDING*2) / diff;
    const float radius = nodeSize / 2;

    
    constexpr float OUTLINE = 0.3f;

    // constexpr auto BATCH_SIZE = 8;

    size_t batch = data->size();
    for (size_t i = 0; i < batch; i++){
        auto dataPoint = data->operator[](i);
        auto coordinates = dataPoint.input;

        int correctIndex = dataPoint.expect[0] != 0 ?
            0 : 1;
        auto color = correctIndex == 0 ? 
            sf::Color(0x02d309FF) : // green
            sf::Color(0xd31302FF); // red

        auto x = (coordinates[0] - min) * nodeSize + PADDING;
        auto y = (coordinates[1] - min) * nodeSize + PADDING;


        network->learn(dataPoint);
        
        auto networkGuessColor = 
            network->classify() == correctIndex ?
                sf::Color(0x85f78dFF) : //  correct
                sf::Color(0xf78f85FF); // invalid

        drawNode(
            window, radius*(1.0f + 2*OUTLINE),
            x - radius*OUTLINE,
            y - radius*OUTLINE,
            networkGuessColor
        );

        drawNode(
            window, radius, 
            x, y, color
        );
    }

    sf::Font font;
    font.loadFromFile("Ubuntu-L.ttf");
    sf::Text text(
        "FPS: " + std::to_string(1000.0f / float(time)) +
        " Loss: " + std::to_string(network->cost()) +
        " " + std::to_string(network->_average_loss),
        font, 24
    );
    network->reset_loss();
    text.setPosition({0,0});
    text.setFillColor(sf::Color::White);
    text.setOutlineColor(sf::Color::White);

    // network->apply(batch);

    window->draw(text);

    window->display();
}

void neuralNetworkPointTest(){
    using namespace sf;

    auto window = RenderWindow(VideoMode(1000, 800), "CLife");

    neural_network::NeuralNetwork net({2, 5, 2}, new ReLu());
        
    window.display();

    constexpr double MIN = 0, MAX = 100;

    test_creator::TestCreator creator;
    std::unique_ptr<std::vector<data::Data>> data(
        creator.prepare([](double x, double y){
            return x*x + y*y <= MAX*MAX*0.5f;
    }).createPointTest(MIN, MAX, 512));

    unsigned long long batchSize = 0;

    Clock timer;

    Clock displayTimer;
    while(window.isOpen()){
        Event event; 
        if(window.pollEvent(event)){
            switch (event.type)
            {
            case Event::Closed:{
                window.close();
                db::FileManager fm("networkRL1.txt");
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
            neuralNetworkPointTestVisalization(&window, &net, data.get(), MIN, MAX, time);
            batchSize += data->size();
            if (timer.getElapsedTime().asMilliseconds() >= 500){
                printf("Apply batch: %llu\n", batchSize);
                net.apply(batchSize);
                timer.restart();
                batchSize = 0;
            }

        }
    }
    return;
}

void windowWithDrawer(){
    neuralNetworkPointTest();
    // baseWindowPlayer(neuralNetworkVisualization);
}

END_NAMESPACE