#include <ui/ui.hpp>
#include <core/OLayer.hpp>

int main()
{
    neural_network::OLayer o(3, 5);
    o.initialize().calc_activations({1,2,3,4});
    ui::windowWithDrawer();
    return 0;
}