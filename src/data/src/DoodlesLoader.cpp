#include <data/DoodlesLoader.hpp>


START_NAMESPACE_DATA

DoodlesLoader& DoodlesLoader::load(const std::string& path){
    _path = path;
    return *this;
}

matrix_t* DoodlesLoader::get_images(){
    std::ifstream file(_path);
    
    if(!file.is_open()){
        throw std::runtime_error("File not found: " + _path);
    }
    
    std::string line;
    
    matrix_t* images = new matrix_t();

    long long int key_id = 0;
    std::string country_code;
    bool recognized = false;
    uint32_t timestamp = 0;
    uint16_t n_strokes = 0;
    // uint8_t keys[8];
    file.read(reinterpret_cast<char*>(&key_id), 8);
    file.read(country_code.data(), 2);
    file.read(reinterpret_cast<char*>(&recognized), sizeof(recognized));
    file.read(reinterpret_cast<char*>(&timestamp), sizeof(timestamp));
    file.read(reinterpret_cast<char*>(&n_strokes), sizeof(n_strokes));

    printf("key_id: %lld\n", key_id);
    printf("country_code: %s\n", country_code.data());
    printf("recognized: %d\n", recognized);
    printf("timestamp: %d\n", timestamp);
    printf("n_strokes: %d\n", n_strokes);

    uint16_t n_points;
    std::vector<double> pixels(256*256, 0.0);
    
    for (uint16_t n = 0; n < n_strokes; n++){
        file.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
        std::vector<uint8_t> x(n_points, 0);
        std::vector<uint8_t> y(n_points, 0);

        file.read(reinterpret_cast<char*>(x.data()), n_points * sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(y.data()), n_points * sizeof(uint8_t));

        printf("n_points: %d\n", n_points);
        for (uint16_t i = 0; i < n_points; i++){
            uint8_t pixel_x = x[i];
            uint8_t pixel_y = y[i];
            printf("%i. %d %d\n", i, x, y);
            pixels[pixel_x + pixel_y*256] = 1.0;
        }
    }
    images->push_back(pixels);

    // for (int i = 0; i < 8; i++){
    //     printf("%d\n", keys[i]);
    // }
    


    return images;
}

END_NAMESPACE