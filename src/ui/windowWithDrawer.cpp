#include "windowWithDrawer.hpp"

START_NAMESPACE_UI

void windowWithDrawer(){
    using namespace sf;

    auto window = RenderWindow(VideoMode(1000, 800), "CLife");
        
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
    }
    return;
}


END_NAMESPACE