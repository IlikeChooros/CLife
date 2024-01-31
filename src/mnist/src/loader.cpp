#include <mnist/loader.hpp>

START_NAMESPACE_MNIST

Loader::Loader(const std::string& path){
    load(path);
}

Loader& Loader::load(const std::string& path){
    this->path = path;
    return *this;
}

data::matrix_t* Loader::get_images(){
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file: " + path);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    // Convert from big endian to little endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    n_rows = __builtin_bswap32(n_rows);
    n_cols = __builtin_bswap32(n_cols);

    if(magic_number != MNIST_MAGIC_NUMBER){
        throw std::runtime_error("Invalid MNIST image file!");
    }

    int image_size = n_rows * n_cols;
    std::unique_ptr<std::vector<std::vector<double>>> images(
        new std::vector<std::vector<double>>(number_of_images, std::vector<double>(image_size))
    );
    

    for(int i = 0; i < number_of_images; i++){
        std::vector<uint8_t> image(image_size);
        file.read(reinterpret_cast<char*>(image.data()), image_size);

        std::transform(image.begin(), image.end(), (*images)[i].begin(), [](uint8_t pixel){
            return static_cast<double>(pixel) / 255.0f; // Normalize to [0, 1]
        });
        std::cout << std::endl;
    }

    return images.release();
}

/**
 * @brief Get the labels object
*/
data::matrix_t* Loader::get_labels(){
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file: " + path);
    }

    int32_t magic_number = 0;
    int32_t num_labels = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    // Convert from big endian to little endian
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    if(magic_number != MNIST_LABEL_MAGIC_NUMBER){
        throw std::runtime_error("Invalid MNIST label file!");
    }

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    std::unique_ptr<std::vector<std::vector<double>>> labels_double(
        new std::vector<std::vector<double>>(num_labels, std::vector<double>(10, 0))
    );
    for(int i = 0; i < num_labels; i++){
        (*labels_double)[i][labels[i]] = 1;
    }

    return labels_double.release();
}

data::data_batch* Loader::merge_data(
    data::matrix_t* images,
    data::matrix_t* labels
){
    if(images->size() != labels->size()){
        throw std::runtime_error("Number of images and labels does not match!");
    }

    std::unique_ptr<data::data_batch> data(new data::data_batch(images->size()));
    for(size_t i = 0; i < images->size(); i++){
        (*data)[i].input = (*images)[i];
        (*data)[i].expect = (*labels)[i];
    }

    return data.release();
}

END_NAMESPACE_MNIST