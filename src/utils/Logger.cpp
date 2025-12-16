/*******************************************************************************
 * CinematicFX - Logger Implementation
 * 
 * Debug and performance logging
 ******************************************************************************/

#include "Logger.h"
#include <cstdio>
#include <cstdarg>
#include <ctime>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Undefine Windows macros that conflict with our enum
#ifdef ERROR
#undef ERROR
#endif
#ifdef DEBUG
#undef DEBUG
#endif
#ifdef INFO
#undef INFO
#endif
#endif

namespace CinematicFX {

// Static members
Logger::LogLevel Logger::current_level_ = LogLevel::INFO;
FILE* Logger::log_file_ = nullptr;
bool Logger::console_output_ = true;

void Logger::Initialize(LogLevel level, const char* log_file_path) {
    current_level_ = level;
    
    if (log_file_path && log_file_path[0] != '\0') {
        log_file_ = fopen(log_file_path, "a");
        if (!log_file_) {
            fprintf(stderr, "Failed to open log file: %s\n", log_file_path);
        }
    }
}

void Logger::Shutdown() {
    if (log_file_) {
        fclose(log_file_);
        log_file_ = nullptr;
    }
}

void Logger::SetLevel(LogLevel level) {
    current_level_ = level;
}

void Logger::SetConsoleOutput(bool enabled) {
    console_output_ = enabled;
}

void Logger::Debug(const char* format, ...) {
    if (current_level_ > LogLevel::DEBUG) return;
    
    va_list args;
    va_start(args, format);
    Log(LogLevel::DEBUG, format, args);
    va_end(args);
}

void Logger::Info(const char* format, ...) {
    if (current_level_ > LogLevel::INFO) return;
    
    va_list args;
    va_start(args, format);
    Log(LogLevel::INFO, format, args);
    va_end(args);
}

void Logger::Warning(const char* format, ...) {
    if (current_level_ > LogLevel::WARNING) return;
    
    va_list args;
    va_start(args, format);
    Log(LogLevel::WARNING, format, args);
    va_end(args);
}

void Logger::Error(const char* format, ...) {
    if (current_level_ > LogLevel::ERROR) return;
    
    va_list args;
    va_start(args, format);
    Log(LogLevel::ERROR, format, args);
    va_end(args);
}

void Logger::Log(LogLevel level, const char* format, va_list args) {
    // Get timestamp
    time_t now = time(nullptr);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    // Get level string
    const char* level_str = "UNKNOWN";
    switch (level) {
        case LogLevel::DEBUG:   level_str = "DEBUG"; break;
        case LogLevel::INFO:    level_str = "INFO "; break;
        case LogLevel::WARNING: level_str = "WARN "; break;
        case LogLevel::ERROR:   level_str = "ERROR"; break;
    }
    
    // Format message
    char message[1024];
    vsnprintf(message, sizeof(message), format, args);
    
    // Full log line
    char log_line[1200];
    snprintf(log_line, sizeof(log_line), "[%s] [%s] %s\n", timestamp, level_str, message);
    
    // Console output
    if (console_output_) {
#ifdef _WIN32
        OutputDebugStringA(log_line);
#endif
        fprintf((level >= LogLevel::ERROR) ? stderr : stdout, "%s", log_line);
    }
    
    // File output
    if (log_file_) {
        fprintf(log_file_, "%s", log_line);
        fflush(log_file_);
    }
}

} // namespace CinematicFX
