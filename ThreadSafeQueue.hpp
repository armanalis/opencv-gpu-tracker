#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

//using template in order to change inside of it like cv::Mat or int
template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    std::mutex mutex; // prevent race conditions
    std::condition_variable cond;

public:
    void push(T item) {
        //lock mutex
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(item);
        // queue the data and wake the thread.
        cond.notify_one(); 
    } // when curly bracket ends the lock is unlocked.(lock_guard feature)

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        // if the queue is empty, sleep this thread in order to overload the CPU.
        cond.wait(lock, [this]() { return !queue.empty(); });
        
        item = queue.front();
        queue.pop();
        return true;
    }
};