#include <ui/windowWithDrawer.hpp>

START_NAMESPACE_UI

void drawNode(sf::RenderWindow *window, float radius, float posX, float posY, const sf::Color &color = sf::Color(250, 100, 100))
{
    sf::CircleShape node(radius);

    node.setPosition({posX, posY});
    node.setFillColor(color);

    window->draw(node);
}

float getStartY(float height, float maxElements, float n)
{
    return height * (1 - n / maxElements) / 2;
}

void renderNetworkGuess(
    sf::RenderWindow *window,
    neural_network::ONeural *network,
    neural_network::data_batch *data,
    double min,
    double max)
{
    constexpr auto radius = 16.0f;

    sf::Vector2u windowSize = window->getSize();

    auto minSize = std::min(windowSize.x, windowSize.y);

    for (size_t i = 0; i < data->size(); i++)
    {
        auto dataPoint = (*data)[i];
        network->input(dataPoint);
        (void)network->outputs();
        auto color = network->correct()
                         ? sf::Color(0x02d30940)  // green
                         : sf::Color(0xd3130240); // red

        auto x = ((dataPoint.input[0] - min) / (max - min)) * minSize - radius / 2;
        auto y = ((dataPoint.input[1] - min) / (max - min)) * minSize - radius / 2;

        drawNode(window, radius, x, y, color);
    }
}

void renderPoints(
    sf::RenderWindow *window,
    std::vector<data::Data> *data,
    neural_network::ONeural *network,
    double min,
    double max)
{
    constexpr auto PADDING = 20;
    constexpr auto radius = 5.0f;

    sf::Vector2u windowSize = window->getSize();

    auto minSize = std::min(windowSize.x, windowSize.y);
    minSize -= 2 * PADDING;

    for (size_t i = 0; i < data->size(); i++)
    {
        auto dataPoint = (*data)[i];
        auto color = dataPoint.expect[0] == 1
                         ? sf::Color(0x5BAFFCFF)  // green
                         : sf::Color(0xFD4F5AFF); // red

        auto x = dataPoint.input[0] * minSize + PADDING;
        auto y = (1 - dataPoint.input[1]) * minSize + PADDING;

        drawNode(window, radius, x, y, color);
    }

    constexpr int pixelSize = 4;

    sf::VertexArray pixels(sf::Triangles, windowSize.x * windowSize.y / (pixelSize * pixelSize) * 6);

    for (int y = 0; y < windowSize.y; y += pixelSize)
    {
        for (int x = 0; x < windowSize.x; x += pixelSize)
        {
            auto fx = float(x) / float(windowSize.x);
            auto fy = float(y) / float(windowSize.y);

            network->raw_input({fx, fy});

            auto netColor = network->classify() == 0
                                ? sf::Color(0x5BAFFC40)  // blue
                                : sf::Color(0xFD4F5A40); // red

            auto triangles = &pixels[(x / pixelSize + (y / pixelSize) * (windowSize.x / pixelSize)) * 6];

            // triangles[0] = sf::Vertex(sf::Vector2f(x, windowSize.y - y - pixelSize), netColor);
            // triangles[1] = sf::Vertex(sf::Vector2f(x, windowSize.y - y), netColor);
            // triangles[2] = sf::Vertex(sf::Vector2f(x + pixelSize, windowSize.y - y), netColor);
            // triangles[3] = sf::Vertex(sf::Vector2f(x + pixelSize, windowSize.y - y - pixelSize), netColor);

            triangles[0] = sf::Vertex(sf::Vector2f(x, windowSize.y - y - pixelSize), netColor);
            triangles[1] = sf::Vertex(sf::Vector2f(x + pixelSize, windowSize.y - y), netColor);
            triangles[2] = sf::Vertex(sf::Vector2f(x + pixelSize, windowSize.y - y - pixelSize), netColor);

            triangles[3] = sf::Vertex(sf::Vector2f(x, windowSize.y - y), netColor);
            triangles[4] = sf::Vertex(sf::Vector2f(x + pixelSize, windowSize.y - y), netColor);
            triangles[5] = sf::Vertex(sf::Vector2f(x, windowSize.y - y - pixelSize), netColor);
        }
    }
    window->draw(pixels);
}

void neuralNetworkPointTest()
{
    using namespace sf;

    auto window = RenderWindow(VideoMode(512, 512), "CLife");

    neural_network::ONeural net = neural_network::ONeural(
        {2, 6, 2}, ActivationType::softmax, ActivationType::relu);
    net.initialize();
    window.display();

    constexpr double MIN = 0, MAX = 512;

    test_creator::TestCreator creator;
    std::unique_ptr<data::data_batch> data(
        creator.prepare([MAX, MIN](double x, double y)
                        {
                            return -0.5f * x + MAX * 0.6f < y;
                            // return x * x + (y - MAX) * (y - MAX) < MAX * 0.1;
                            // return x*x + y*y - y * 0.3f * MAX - x * 0.5f<= MAX*MAX*0.3f;

                            // return (MAX * 0.000025) * x * x - (MAX * 0.01) * x + (MAX * 1.2) > y;
                            // return y < 100 || y > 300;
                            // return (x - MAX*0.5f)*(x - MAX*0.5f) + (y - MAX*0.5f)*(y - MAX*0.5f) <= MAX*MAX*0.1f;
                            // return 0.0018*x*x - 6*x + MAX*1.2 > y;
                            // return (MAX*0.000025)*x*x - (MAX*0.01)*x + (MAX*1.2)> y;
                        })
            .createPointTest(MIN, MAX, 512));

    Clock timer;

    Clock displayTimer;
    while (window.isOpen())
    {
        Event event;
        if (window.pollEvent(event))
        {
            switch (event.type)
            {
            case Event::Closed:
            {
                window.close();
            }
            break;

            default:
                break;
            }
        }
        auto time = displayTimer.getElapsedTime().asMilliseconds();
        if (time >= 20)
        {
            displayTimer.restart();
            window.clear(sf::Color(0x222222));
            renderPoints(&window, data.get(), &net, MIN, MAX);
            // renderNetworkGuess(&window, &net, data.get(), MIN, MAX);
            net.batch_learn(data.get(), 0.3, 32);
            // net.raw_input({double(Mouse::getPosition(window).x), double(Mouse::getPosition(window).y)});
            window.display();
            // if (timer.getElapsedTime().asMilliseconds() >= 1000)
            // {
            //     timer.restart();
            //     auto dataPoint = (*data)[rand() % data->size()];
            //     net.input(dataPoint);
            //     auto outputs = net.outputs();
            //     printf(
            //         "Cost: %f Loss: %f Output0: %f Output1: %f classify: %lu x: %f y: %f\n",
            //         net.cost(), net.loss(), outputs[0], outputs[1], net.classify(),
            //         dataPoint.input[0] * (MAX - MIN), dataPoint.input[1] * (MAX - MIN));
            // }
        }
    }
    return;
}

void windowWithDrawer()
{
    srand(time(NULL));
    neuralNetworkPointTest();
}

END_NAMESPACE