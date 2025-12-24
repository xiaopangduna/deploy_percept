#ifndef DEPLOY_PERCEPT_POST_PROCESS_TYPES_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_TYPES_HPP

#include <cstdint>

namespace deploy_percept {
namespace post_process {

struct BoxRect {
    int left;
    int right;
    int top;
    int bottom;
};

struct DetectResult {
    char name[16];  // OBJ_NAME_MAX_SIZE
    BoxRect box;
    float prop;
};

struct DetectResultGroup {
    int id;
    int count;
    DetectResult results[64];  // OBJ_NUMB_MAX_SIZE
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_TYPES_HPP