#include <ui/Drawer.hpp>

START_NAMESPACE_UI

void do_nothing(neural_network::vector_t){return;}

Drawer::Drawer(
    size_t width, size_t height,
    size_t pixelRows, size_t pixelCols,
    bool monochromatic
): 
width(width), height(height), 
pixelRows(pixelRows), pixelCols(pixelCols), 
_monochromatic(monochromatic), _callback(do_nothing)
{
    _pixels.assign(3, std::vector<std::vector<uint8_t>>(pixelRows, std::vector<uint8_t>(pixelCols, 0)));
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
        int x = i/pixelCols;
        int y = i%pixelCols;
        double R = pixels[i] * 255.0, G = R, B = R;
    
        if (!_monochromatic){
            if (abs(R) > 255){
                R = 255;
            }

            if (R < 0){
                R = abs(R);
                G = 0;
                B = 0;
            }
            else{
                G = R;
                B = R;
                R = 0;
            }
        }

        _pixels[0][x][y] = (uint8_t)R;
        _pixels[1][x][y] = (uint8_t)G;
        _pixels[2][x][y] = (uint8_t)B;
    }

    return *this;
}

neural_network::vector_t Drawer::getPixels(){
    neural_network::vector_t pixels(pixelRows*pixelCols, 0);
    for(size_t i = 0; i < pixelRows; i++){
        for(size_t j = 0; j < pixelCols; j++){
            pixels[i*pixelCols + j] = static_cast<neural_network::real_number_t>(_pixels[0][i][j]) / 255.0;
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
            pixel.setFillColor(sf::Color(_pixels[0][i][j], _pixels[1][i][j], _pixels[2][i][j]));
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
            int nx = x + i, ny = y + j;

            if(
                nx >= 0 && nx < static_cast<int>(pixelCols) &&
                ny >= 0 && ny < static_cast<int>(pixelRows)
            ){
                if(erase){
                    _pixels[0][ny][nx] = 0;
                    _pixels[1][ny][nx] = 0;
                    _pixels[2][ny][nx] = 0;
                } 
                else {
                    _pixels[0][ny][nx] = std::min<int>(
                        255, 
                        _pixels[0][ny][nx] + 255 * (1 - static_cast<float>(std::abs(i) + std::abs(j)) / static_cast<float>(radius * 2 + 1))
                    );
                    _pixels[1][ny][nx] = _pixels[0][ny][nx];
                    _pixels[2][ny][nx] = _pixels[0][ny][nx];
                }
            }
        }
    }
}

END_NAMESPACE