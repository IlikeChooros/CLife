#include <ui/Drawer.hpp>

START_NAMESPACE_UI

void do_nothing(neural_network::vector_t){return;}

Drawer::Drawer(
    size_t width, size_t height,
    size_t pixelRows, size_t pixelCols
): width(width), height(height),pixelRows(pixelRows),pixelCols(pixelCols), _callback(do_nothing)
{
    _pixels.assign(pixelRows, std::vector<uint8_t>(pixelCols, 0));
    window.create(
        sf::VideoMode(width, height),
        "Neural Network Drawer",
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);
}

Drawer& Drawer::setCallback(std::function<void(neural_network::vector_t)> callback){
    _callback = callback;
    return *this;
}

bool Drawer::closed(){
    return !window.isOpen();
}

void Drawer::open(){
    bool hold = false;
    bool erase = false;
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
                hold = true;
                erase = event.mouseButton.button == sf::Mouse::Right;
                break;
            case sf::Event::MouseButtonReleased:
                hold = false;
                break;
            default:
                break;
            }
        }
        if(hold){
            auto position = sf::Mouse::getPosition(window);
            sf::Event::MouseButtonEvent event;
            event.x = position.x;
            event.y = position.y;
            _draw(event, erase);
            _callback(getPixels());
        }
        _drawPixels();
        window.display();
    }
}

Drawer& Drawer::loadPixels(const neural_network::vector_t& pixels){
    if (pixels.size() != pixelCols*pixelRows){
        return *this;
    }

    for(size_t i = 0; i < pixels.size(); i++){
        _pixels[i/pixelCols][i%pixelCols] = static_cast<uint8_t>(pixels[i] * 255.0);
    }

    return *this;
}

neural_network::vector_t Drawer::getPixels(){
    neural_network::vector_t pixels(pixelRows*pixelCols, 0);
    for(size_t i = 0; i < pixelRows; i++){
        for(size_t j = 0; j < pixelCols; j++){
            pixels[i*pixelCols + j] = static_cast<neural_network::real_number_t>(_pixels[i][j]) / 255.0;
        }
    }
    return pixels;
}

void Drawer::_drawPixels(){
    sf::RectangleShape pixel(sf::Vector2f(width/pixelCols, height/pixelRows));
    for(size_t i = 0; i < pixelRows; i++){
        for(size_t j = 0; j < pixelCols; j++){
            pixel.setPosition(
                j * width/pixelCols,
                i * height/pixelRows
            );
            pixel.setFillColor(sf::Color(_pixels[i][j], _pixels[i][j], _pixels[i][j]));
            pixel.setOutlineColor(sf::Color::White);
            pixel.setOutlineThickness(1);
            window.draw(pixel);
        }
    }

}

void Drawer::_draw(sf::Event::MouseButtonEvent& event, bool erase){
    int x = static_cast<float>(event.x) / (static_cast<float>(height)/pixelCols);
    int y = static_cast<float>(event.y) / (static_cast<float>(height)/pixelRows);

    constexpr auto BRUSH_RADIUS = 1, ERASE_RADIUS = 2;

    const auto radius = erase ? ERASE_RADIUS : BRUSH_RADIUS;

    for(int i = -radius; i <= radius; i++){
        for(int j = -radius; j <= radius; j++){
            if(
                x + i >= 0 && x + i < static_cast<int>(pixelCols) &&
                y + j >= 0 && y + j < static_cast<int>(pixelRows)
            ){
                if(erase){
                    _pixels[y + j][x + i] = 0;
                } 
                else {
                    _pixels[y + j][x + i] = std::min<int>(
                        255, 
                        _pixels[y + j][x + i] + 255 * (1 - static_cast<float>(std::abs(i) + std::abs(j)) / static_cast<float>(radius * 2 + 1))
                    );
                }
            }
        }
    }
}

END_NAMESPACE