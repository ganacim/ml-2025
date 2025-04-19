#ifndef _TIMER_H
#define _TIMER_H

#include <string>
#include <vector>
#include <cuda_runtime.h>  
namespace util {

    class base_timer {
    public:
        virtual void do_start() = 0;
        virtual void do_stop() = 0;
        virtual float do_get_elapsed() const = 0;
        virtual ~base_timer() = default;
    };

    class cpu_timer : public base_timer {
    public:
        cpu_timer();
        ~cpu_timer() override;

        void do_start() override;
        void do_stop() override;
        float do_get_elapsed() const override;

        double get_cpu_time() const;
        bool is_stopped() const;

    private:
        double m_start_time{0};
        double m_stop_time{0};
        bool m_is_stopped{true};
    };

    class gpu_timer : public base_timer {
    public:
        gpu_timer();
        ~gpu_timer() override;

        void do_start() override;
        void do_stop() override;
        float do_get_elapsed() const override;

    private:
        cudaEvent_t m_start;  
        cudaEvent_t m_stop;
    };

    class scoped_timer_stop {
    public:
        explicit scoped_timer_stop(base_timer& timer);
        void stop();

    private:
        base_timer* m_timer;
    };

    class timer_pool {
    public:
        timer_pool();
        ~timer_pool();

        gpu_timer& gpu_add(const std::string& label, size_t data_size);
        cpu_timer& cpu_add(const std::string& label, size_t data_size);

        void flush();

    private:
        std::vector<base_timer*> m_timers;
    };

} // namespace util

#endif // TIMER_H
