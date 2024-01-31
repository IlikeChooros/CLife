#pragma once


#include <fstream>

#include "Data.hpp"

START_NAMESPACE_DATA


class DoodlesLoader
{
    std::string _path;
public:
    DoodlesLoader() = default;
    ~DoodlesLoader() = default;

    DoodlesLoader& load(const std::string& path);
    matrix_t* get_images();
    matrix_t* get_labels();

    data_batch* merge_data(matrix_t* images, matrix_t* labels);
};


END_NAMESPACE