/*******************************************************************************
 * CinematicFX - Performance Timer
 ******************************************************************************/

#pragma once

#include <chrono>

namespace CinematicFX {

    /**
     * @brief High-resolution performance timer
     */
    class PerformanceTimer {
    public:
        PerformanceTimer() : running_(false) {}
        
        /**
         * @brief Start the timer
         */
        void Start() {
            start_time_ = std::chrono::high_resolution_clock::now();
            running_ = true;
        }
        
        /**
         * @brief Stop the timer
         */
        void Stop() {
            end_time_ = std::chrono::high_resolution_clock::now();
            running_ = false;
        }
        
        /**
         * @brief Get elapsed time in milliseconds
         * @return Elapsed time (auto-stops if still running)
         */
        float ElapsedMs() {
            if (running_) {
                Stop();
            }
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time_ - start_time_
            );
            
            return duration.count() / 1000.0f;
        }
        
        /**
         * @brief Get elapsed time in seconds
         */
        float ElapsedSeconds() {
            return ElapsedMs() / 1000.0f;
        }
        
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point end_time_;
        bool running_;
    };

} // namespace CinematicFX
