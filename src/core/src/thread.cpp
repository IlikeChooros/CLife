#include <core/thread.hpp>


START_NAMESPACE_NEURAL_NETWORK

ThreadPool::ThreadPool(size_t threads) : _stop(false){
    for(size_t i = 0; i < threads; i++){
        _workers.emplace_back(
            [this](){
                while(true){
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->_queue_mutex);
                        this->_condition.wait(
                            lock, 
                            [this](){return this->_stop || !this->_tasks.empty();}
                        );
                        if(this->_stop && this->_tasks.empty()){
                            return;
                        }
                        task = std::move(this->_tasks.front());
                        this->_tasks.pop();
                    }
                    task();
                }
            }
        );
    }
}

ThreadPool::~ThreadPool(){
  execute();
}

void ThreadPool::enqueue(std::function<void()> task){
    {
        std::unique_lock<std::mutex> lock(_queue_mutex);
        _tasks.emplace(task);
    }
    _condition.notify_one();
}

void ThreadPool::execute(){
  if(_stop) return; // if the thread pool is stopped, return
  {
      std::unique_lock<std::mutex> lock(_queue_mutex);
      _stop = true;
  }
  _condition.notify_all();
  for(auto& worker : _workers){
      worker.join();
  }
}

END_NAMESPACE