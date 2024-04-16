#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <functional>

#include "namespaces.hpp"

START_NAMESPACE_NEURAL_NETWORK

class ThreadPool{
    public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    /**
     * @brief Add a task to the thread pool
     * @param task task to be added, note that it doesn't take any arguments 
    */
    void enqueue(std::function<void()> task);

    /**
     * @brief Execute all the tasks in the thread pool
    */
    void execute();

    private:
    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;

    std::mutex _queue_mutex;
    std::condition_variable _condition;
    bool _stop;
};

END_NAMESPACE