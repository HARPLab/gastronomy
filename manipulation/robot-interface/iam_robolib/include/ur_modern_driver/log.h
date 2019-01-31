#pragma once
#include <inttypes.h>

#define LOG_DEBUG(format, ...) printf("[DEBUG]: " format "\n", ##__VA_ARGS__)
#define LOG_WARN(format, ...) printf("[WARNING]: " format "\n", ##__VA_ARGS__)
#define LOG_INFO(format, ...) printf("[INFO]: " format "\n", ##__VA_ARGS__)
#define LOG_ERROR(format, ...) printf("[ERROR]: " format "\n", ##__VA_ARGS__)
#define LOG_FATAL(format, ...) printf("[FATAL]: " format "\n", ##__VA_ARGS__)
