#ifndef DEPLOY_PERCEPT_POST_PROCESS_TYPES_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_TYPES_HPP

#include <cstdint>
#include <cstring>

namespace deploy_percept {
namespace post_process {

struct BoxRect {
    int left;
    int right;
    int top;
    int bottom;
    
    // 默认构造函数，初始化所有成员为0
    BoxRect() : left(0), right(0), top(0), bottom(0) {}
    
    // 聚合初始化构造函数，保持向后兼容性
    BoxRect(int l, int r, int t, int b) : left(l), right(r), top(t), bottom(b) {}
};

struct DetectResult {
    char name[16];  // OBJ_NAME_MAX_SIZE
    BoxRect box;
    float prop;
    
    // 默认构造函数，初始化所有成员
    DetectResult() : box(), prop(0.0f) {
        memset(name, 0, sizeof(name));  // 将字符数组清零
    }
    
    // 聚合初始化构造函数
    DetectResult(const char* n, BoxRect b, float p) : box(b), prop(p) {
        if (n) {
            strncpy(name, n, sizeof(name) - 1);
            name[sizeof(name) - 1] = '\0';
        } else {
            memset(name, 0, sizeof(name));
        }
    }
};

struct DetectResultGroup {
    int id;
    int count;
    DetectResult results[64];  // OBJ_NUMB_MAX_SIZE
    
    // 默认构造函数，初始化所有成员
    DetectResultGroup() : id(0), count(0), results{} {}
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_TYPES_HPP