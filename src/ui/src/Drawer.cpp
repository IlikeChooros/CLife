#include <ui/Drawer.hpp>

START_NAMESPACE_UI

Drawer::Drawer(
    size_t width, size_t height,
    size_t pixelsWidth, size_t pixelsHeight
): width(width), height(height),
   pixelWidth(pixelsWidth), pixelHeight(pixelsHeight)
{
    window.create(
        sf::VideoMode(width, height, 8),
        "Neural Network Drawer",
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);
}

void Drawer::open(){
    
    while(window.isOpen()){
        sf::Event event;
        window.clear(sf::Color::Black);
        while(window.pollEvent(event)){
            switch (event.type)
            {
            case sf::Event::Closed:
                window.close();
                break;
            case sf::Event::MouseButtonPressed:
                _draw(event.mouseButton);
                break;
            default:
                break;
            }
        }
        
        window.display();
    }
}

void Drawer::_draw(sf::Event::MouseButtonEvent& event){
    sf::RectangleShape pixel(sf::Vector2f(width/pixelWidth, height/pixelHeight));
    pixel.setPosition(
        event.x,
        event.y
    );
    pixel.setFillColor(event.button == sf::Mouse::Left ? sf::Color::White : sf::Color::Black);
    window.draw(pixel);
}

END_NAMESPACE