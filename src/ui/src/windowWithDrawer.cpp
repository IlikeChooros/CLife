#include <ui/windowWithDrawer.hpp>

START_NAMESPACE_UI



void drawNode(sf::RenderWindow* window, float radius, float posX, float posY, const sf::Color& color = sf::Color(250, 100 ,100)){
    sf::CircleShape node(radius);

    node.setPosition({posX, posY});
    node.setFillColor(color);

    window->draw(node);
}

float getStartY(float height, float maxElements, float n){
    return height * (1 - n / maxElements) / 2;
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
        {2,10,2}, ActivationType::softmax, ActivationType::relu);
    net.initialize();
    window.display();

    constexpr double MIN = 0, MAX = 400;

    test_creator::TestCreator creator;
    std::unique_ptr<data::data_batch> data(
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
            net.batch_learn(data.get(), 0.6, 64);
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
        
        
    }
    return;
}

void windowWithDrawer(){
    srand(time(NULL));
    neuralNetworkPointTest();
}

END_NAMESPACE