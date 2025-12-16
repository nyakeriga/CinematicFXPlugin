/*******************************************************************************
 * CinematicFX - Logger Header
 ******************************************************************************/

#pragma once

#include <cstdarg>
#include <cstdio>

namespace CinematicFX {

    /**
     * @brief Logging utility for debug and performance monitoring
     */
    class Logger {
    public:
        enum class LogLevel {
            DEBUG = 0,
            INFO = 1,
            WARNING = 2,
            ERROR = 3,
            NONE = 4
        };
        
        /**
         * @brief Initialize logger
         * @param level Minimum log level
         * @param log_file_path Optional log file path (nullptr for no file logging)
         */
        static void Initialize(LogLevel level = LogLevel::INFO, const char* log_file_path = nullptr);
        
        /**
         * @brief Shutdown logger and close log file
         */
        static void Shutdown();
        
        /**
         * @brief Set minimum log level
         */
        static void SetLevel(LogLevel level);
        
        /**
         * @brief Enable/disable console output
         */
        static void SetConsoleOutput(bool enabled);
        
        /**
         * @brief Log debug message
         */
        static void Debug(const char* format, ...);
        
        /**
         * @brief Log info message
         */
        static void Info(const char* format, ...);
        
        /**
         * @brief Log warning message
         */
        static void Warning(const char* format, ...);
        
        /**
         * @brief Log error message
         */
        static void Error(const char* format, ...);
        
    private:
        static LogLevel current_level_;
        static FILE* log_file_;
        static bool console_output_;
        
        static void Log(LogLevel level, const char* format, va_list args);
    };

} // namespace CinematicFX
