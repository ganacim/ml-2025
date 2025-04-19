#include <iostream>
#include <sstream>
#include <cmath>
#include <stack>
#include <iomanip>

#include "error.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <chrono>



#ifdef _WIN32
#include <windows.h>
#else
#include <ctime>
#include <sys/time.h>
#endif

#ifdef _MSC_VER
#	include <float.h>
#	define isinf _finite
#	define isnan _isnan
#endif

namespace util {    
    timer_pool timers;
    
    

    
    cpu_timer::cpu_timer() : m_is_stopped(true) {}

    cpu_timer::~cpu_timer() {}

    void cpu_timer::do_start() {
        m_start_time = get_cpu_time();
        m_is_stopped = false;
    }

    void cpu_timer::do_stop() {
        m_stop_time = get_cpu_time();
        m_is_stopped = true;
    }

    float cpu_timer::do_get_elapsed() const {
        if (is_stopped()) {
            return static_cast<float>(m_stop_time - m_start_time);
        } else {
            return static_cast<float>(get_cpu_time() - m_start_time);
        }
    }

    double cpu_timer::get_cpu_time() const {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    bool cpu_timer::is_stopped() const {
        return m_is_stopped;
    }

    
    gpu_timer::gpu_timer() {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }

    gpu_timer::~gpu_timer() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

    void gpu_timer::do_start() {
        cudaEventRecord(m_start, 0);
    }

    void gpu_timer::do_stop() {
        cudaEventRecord(m_stop, 0);
        cudaEventSynchronize(m_stop);
    }

    float gpu_timer::do_get_elapsed() const {
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, m_start, m_stop);
        return elapsed_time / 1000.0f;  
    }

    
    scoped_timer_stop::scoped_timer_stop(base_timer& timer) : m_timer(&timer) {}

    void scoped_timer_stop::stop() {
        m_timer->do_stop();
    }

    
    timer_pool::timer_pool() = default;

    timer_pool::~timer_pool() = default;

    gpu_timer& timer_pool::gpu_add(const std::string& label, size_t data_size) {
        auto* timer = new gpu_timer();
        m_timers.push_back(timer);
        return *timer;
    }

    cpu_timer& timer_pool::cpu_add(const std::string& label, size_t data_size) {
        auto* timer = new cpu_timer();
        m_timers.push_back(timer);
        return *timer;
    }

    void timer_pool::flush() {
        for (auto* timer : m_timers) {
            timer->do_stop();
        }
        m_timers.clear();
    }

} 