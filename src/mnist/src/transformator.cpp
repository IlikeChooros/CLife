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
    
    constexpr float RADIANS = 0.0174532925f;

    std::unique_ptr<data::data_batch> noisy(
        new data::data_batch(*data));

    int size = noisy->size();
    int x, y;

    for (int i = 0; i < size; ++i){
        // get random x and y, the movement vector
        x = dist(engine);
        y = dist(engine);
        
        // copy the pixels
        auto pixels = (*noisy)[i].input;
        // move the pixels
        data::vector_t new_pixels = move(pixels, cols, rows, x, y);

        // add random rotation
        // std::uniform_int_distribution center_dist(cols / 2 - 2, cols / 2 + 2);
        std::uniform_real_distribution angle_dist(-40.0f, 40.0f);
        new_pixels = rotate(
            new_pixels, cols, rows, angle_dist(engine) * RADIANS
            // ,center_dist(engine), center_dist(engine)
        );
        
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

data::vector_t transformator::move(
    data::vector_t& pixels,
    size_t cols,
    size_t rows,
    int x,
    int y
){
    data::vector_t moved(pixels.size(), 0);
    int new_y, new_x;

    for (int r = 0; r < static_cast<int>(rows); ++r){
        for (int c = 0; c < static_cast<int>(cols); ++c){
            new_y = r + y;
            new_x = c + x;

            if (new_y >= 0 && new_y < static_cast<int>(rows) 
             && new_x >= 0 && new_x < static_cast<int>(cols)
            ){
                moved[new_y * cols + new_x] = pixels[r * cols + c];
            }
        }
    }

    return moved;
}


data::vector_t transformator::rotate(
    data::vector_t& pixels,
    size_t cols,
    size_t rows,
    float angle,
    int center_x,
    int center_y
){
    data::vector_t rotated(pixels.size(), 0);
    int x, y;
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);

    for (size_t r = 0; r < rows; ++r){
        for (size_t c = 0; c < cols; ++c){
            x = c - center_x;
            y = r - center_y;

            // rotate around the point
            float new_x = x * cos_angle - y * sin_angle + center_x;
            float new_y = x * sin_angle + y * cos_angle + center_y;

            // Bilinear interpolation
            if (static_cast<int>(new_y) >= 0 && static_cast<int>(new_y) < static_cast<int>(rows) 
             && static_cast<int>(new_x) >= 0 && static_cast<int>(new_x) < static_cast<int>(cols)
            ){
                int x1 = std::floor(new_x), x2 = std::ceil(new_x);
                int y1 = std::floor(new_y), y2 = std::ceil(new_y);

                double r1 = (x2 - new_x) * pixels[y1 * cols + x1] + (new_x - x1) * pixels[y1 * cols + x2];
                double r2 = (x2 - new_x) * pixels[y2 * cols + x1] + (new_x - x1) * pixels[y2 * cols + x2];

                rotated[r * cols + c] = (y2 - new_y) * r1 + (new_y - y1) * r2;
            }
        }
    }

    return rotated;
}

END_NAMESPACE_MNIST