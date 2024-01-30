#include <mnist/transformator.hpp>

START_NAMESPACE_MNIST


data::data_batch* transformator::add_noise(
    data::data_batch* data,
    int max_vector,
    size_t cols,
    size_t rows,
    size_t noisiness
){
    std::uniform_int_distribution<int> dist(-max_vector, max_vector);
    std::uniform_int_distribution<size_t> noise_index_gen(0, rows*cols);
    std::uniform_int_distribution<size_t> number_of_noise_gen(noisiness*0.5, noisiness);
    std::uniform_real_distribution<double> noise(0, 0.6);
    std::default_random_engine engine(std::random_device{}());
    

    std::unique_ptr<data::data_batch> noisy(
        new data::data_batch(*data));

    int size = noisy->size();
    int x, y, new_y, new_x;

    for (int i = 0; i < size; ++i){
        // get random x and y, the movement vector
        x = dist(engine);
        y = dist(engine);
        
        // copy the pixels
        auto pixels = (*noisy)[i].input;
        std::vector<double> new_pixels(pixels.size(), 0);

        // apply the movement vector for every pixel, and write it to the new_pixels
        for (int r = 0; r < static_cast<int>(rows); ++r){
            for (int c = 0; c < static_cast<int>(cols); ++c){
                new_y = r + y;
                new_x = c + x;

                if (new_y >= 0 && new_y < static_cast<int>(rows) 
                 && new_x >= 0 && new_x < static_cast<int>(cols)
                ){
                    new_pixels[new_y * cols + new_x] = pixels[r * cols + c];
                }
            }
        }

        // add random noise
        size_t noises = number_of_noise_gen(engine);
        for (size_t j = 0; j < noises; j++){
            auto idx = noise_index_gen(engine);
            new_pixels[idx] = std::min(1.0, new_pixels[idx] + noise(engine));
        }

        (*noisy)[i].input = new_pixels;
    }
    return noisy.release();
}


END_NAMESPACE_MNIST