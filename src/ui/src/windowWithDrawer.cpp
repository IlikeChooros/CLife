#include <ui/windowWithDrawer.hpp>

START_NAMESPACE_UI

void windowWithDrawer(){
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

        neuralNetworkVisualization(window, net);
    }
    return;
}

void drawNode(sf::RenderWindow* window, float radius, float posX, float posY){
    sf::CircleShape node(radius);

    node.setPosition({posX, posY});
    node.setFillColor(sf::Color(250, 100, 100));

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
        auto neuron = layer->neurons[n];

        drawNode(window, radius, startX + layerIndex * (radius* 2 + SPACE), startY + n*(radius* 2 + SPACE) ); 
    }
}

void neuralNetworkVisualization(
    sf::RenderWindow& window, neural_network::NeuralNetwork& network
){
    window.clear();

    constexpr auto PADDING = 50, SPACE = 30;

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



END_NAMESPACE