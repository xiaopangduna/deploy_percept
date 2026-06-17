#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h>

namespace {

std::string read_hostname() {
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf) - 1) != 0) {
        return "unknown";
    }
    return buf;
}

std::string format_local_time() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64] = {};
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t)) == 0) {
        return "unknown";
    }
    return buf;
}

}  // namespace

int main() {
    std::printf("deploy_percept smoke test (stdlib)\n");
    std::printf("  hostname : %s\n", read_hostname().c_str());
    std::printf("  time     : %s\n", format_local_time().c_str());
    std::printf("  pid      : %d\n", static_cast<int>(getpid()));
    std::printf("OK\n");
    return 0;
}
