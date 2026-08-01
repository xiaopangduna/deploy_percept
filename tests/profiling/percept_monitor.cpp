/**
 * 旁路资源画像：按进程名定位目标，周期性读 /proc 采样，直至目标退出。
 *
 * 用法:
 *   percept_monitor --name <proc> [--interval-ms 100] [--warmup-sec 5]
 *                   [--wait-sec 60] [--out stats.csv]
 *
 * 进程名匹配 /proc/<pid>/cmdline 的 argv[0] basename，或 /proc/<pid>/exe basename
 *（避免 TASK_COMM 15 字符截断）。找到后锁定该 PID 直至退出。
 *
 * ---------------------------------------------------------------------------
 * 监控指标与 /proc 字段（见 read_sample）
 * ---------------------------------------------------------------------------
 *
 * | 输出指标        | 单位/含义              | 读取位置                         | 字段 |
 * |-----------------|------------------------|----------------------------------|------|
 * | cpu_percent     | 进程 CPU 占用率 %      | /proc/<pid>/stat                 | 第 14 字段 utime、第 15 字段 stime |
 * |                 | （两次采样 Δticks 换算）| 单位：clock ticks（sysconf CLK_TCK）|
 * |                 |                        | cpu% = 100 * Δ(utime+stime)/CLK_TCK / Δwall_sec |
 * |                 |                        | 多核上可超过 100%                |
 * | rss_mb          | 当前常驻内存           | /proc/<pid>/status               | VmRSS:（kB → MB） |
 * | hwm_mb / rss_hwm| 峰值常驻内存           | /proc/<pid>/status               | VmHWM:（kB → MB） |
 * | threads         | 线程数                 | /proc/<pid>/status               | Threads: |
 *
 * 生命周期：
 *   --wait-sec 内等到唯一同名进程 → attach 后按 --interval-ms 采样 →
 *   /proc/<pid> 消失则结束并打印 summary（前 --warmup-sec 不计入 summary）。
 *
 * 本工具不采：温度、NPU、整机 CPU、IO。
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace
{

struct Options
{
    std::string name;
    int interval_ms{100};
    double warmup_sec{5.0};
    double wait_sec{60.0};
    std::string out_path;
};

/** 单次 /proc 快照；CPU% 需与上一拍的 cpu_ticks + 墙钟间隔在采样循环里计算。 */
struct ProcSample
{
    bool ok{false};
    std::uint64_t cpu_ticks{0}; // /proc/<pid>/stat: utime + stime（clock ticks）
    double rss_mb{0.0};         // /proc/<pid>/status: VmRSS
    double hwm_mb{0.0};         // /proc/<pid>/status: VmHWM
    int threads{0};             // /proc/<pid>/status: Threads
};

struct Agg
{
    std::uint64_t samples{0};
    double cpu_sum{0.0};
    double cpu_max{0.0};
    double rss_sum{0.0};
    double rss_max{0.0};
    double hwm_max{0.0};
    double threads_sum{0.0};
    int threads_max{0};
};

void print_usage(const char *argv0)
{
    std::fprintf(
        stderr,
        "用法:\n"
        "  %s --name <proc> [--interval-ms 100] [--warmup-sec 5]\n"
        "     [--wait-sec 60] [--out stats.csv]\n"
        "\n"
        "按进程名旁路采样 CPU%% / RSS / threads，直至目标退出。\n",
        argv0);
}

bool parse_positive_int(const char *s, int &out)
{
    char *end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > 1000000)
    {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_nonneg_double(const char *s, double &out)
{
    char *end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(v) || v < 0.0)
    {
        return false;
    }
    out = v;
    return true;
}

bool parse_args(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        auto need = [&](const char *flag) -> const char * {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "错误: %s 需要参数\n", flag);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(a, "--name") == 0)
        {
            const char *v = need("--name");
            if (v == nullptr)
            {
                return false;
            }
            opt.name = v;
        }
        else if (std::strcmp(a, "--interval-ms") == 0)
        {
            const char *v = need("--interval-ms");
            if (v == nullptr || !parse_positive_int(v, opt.interval_ms))
            {
                std::fprintf(stderr, "错误: 无效的 --interval-ms\n");
                return false;
            }
        }
        else if (std::strcmp(a, "--warmup-sec") == 0)
        {
            const char *v = need("--warmup-sec");
            if (v == nullptr || !parse_nonneg_double(v, opt.warmup_sec))
            {
                std::fprintf(stderr, "错误: 无效的 --warmup-sec\n");
                return false;
            }
        }
        else if (std::strcmp(a, "--wait-sec") == 0)
        {
            const char *v = need("--wait-sec");
            if (v == nullptr || !parse_nonneg_double(v, opt.wait_sec))
            {
                std::fprintf(stderr, "错误: 无效的 --wait-sec\n");
                return false;
            }
        }
        else if (std::strcmp(a, "--out") == 0)
        {
            const char *v = need("--out");
            if (v == nullptr)
            {
                return false;
            }
            opt.out_path = v;
        }
        else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "错误: 未知参数: %s\n", a);
            return false;
        }
    }

    if (opt.name.empty())
    {
        std::fprintf(stderr, "错误: 必须指定 --name\n");
        return false;
    }
    return true;
}

std::string basename_of(const std::string &path)
{
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos)
    {
        return path;
    }
    return path.substr(pos + 1);
}

std::string read_cmdline_argv0(pid_t pid)
{
    const std::string path = "/proc/" + std::to_string(pid) + "/cmdline";
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return {};
    }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.empty())
    {
        return {};
    }
    const auto nul = data.find('\0');
    return data.substr(0, nul == std::string::npos ? data.size() : nul);
}

std::string read_exe_basename(pid_t pid)
{
    const std::string link = "/proc/" + std::to_string(pid) + "/exe";
    char buf[4096];
    const ssize_t n = ::readlink(link.c_str(), buf, sizeof(buf) - 1);
    if (n <= 0)
    {
        return {};
    }
    buf[n] = '\0';
    return basename_of(buf);
}

bool name_matches(pid_t pid, const std::string &name)
{
    const std::string argv0 = read_cmdline_argv0(pid);
    if (!argv0.empty() && basename_of(argv0) == name)
    {
        return true;
    }
    const std::string exe = read_exe_basename(pid);
    return !exe.empty() && exe == name;
}

std::vector<pid_t> find_pids_by_name(const std::string &name, pid_t self_pid)
{
    std::vector<pid_t> out;
    DIR *dir = ::opendir("/proc");
    if (dir == nullptr)
    {
        return out;
    }
    while (dirent *ent = ::readdir(dir))
    {
        if (ent->d_name[0] < '1' || ent->d_name[0] > '9')
        {
            continue;
        }
        char *end = nullptr;
        const long pid_l = std::strtol(ent->d_name, &end, 10);
        if (end == ent->d_name || *end != '\0' || pid_l <= 0)
        {
            continue;
        }
        const pid_t pid = static_cast<pid_t>(pid_l);
        if (pid == self_pid)
        {
            continue;
        }
        if (name_matches(pid, name))
        {
            out.push_back(pid);
        }
    }
    ::closedir(dir);
    return out;
}

bool process_alive(pid_t pid)
{
    return fs::exists("/proc/" + std::to_string(pid));
}

/**
 * 读取目标进程当前资源快照。
 *
 * /proc/<pid>/stat（man proc）：
 *   格式: pid (comm) state ppid ... utime stime ...
 *   - 第 14 字段 utime：用户态累计 CPU 时间（clock ticks）
 *   - 第 15 字段 stime：内核态累计 CPU 时间（clock ticks）
 *   comm 可能含空格，故从最后一个 ')' 之后切字段：
 *     after[0]=state(第3字段) … after[11]=utime … after[12]=stime
 *
 * /proc/<pid>/status：
 *   - VmRSS:  当前 RSS（kB）→ rss_mb
 *   - VmHWM:  峰值 RSS（kB）→ hwm_mb
 *   - Threads: 线程数
 */
ProcSample read_sample(pid_t pid)
{
    ProcSample s;
    const std::string base = "/proc/" + std::to_string(pid);

    {
        std::ifstream in(base + "/stat");
        std::string content;
        std::getline(in, content);
        if (content.empty())
        {
            return s;
        }
        const auto rparen = content.rfind(')');
        if (rparen == std::string::npos || rparen + 2 >= content.size())
        {
            return s;
        }
        std::vector<std::string> after;
        std::size_t i = rparen + 2;
        while (i < content.size())
        {
            while (i < content.size() && content[i] == ' ')
            {
                ++i;
            }
            if (i >= content.size())
            {
                break;
            }
            const auto j = content.find(' ', i);
            if (j == std::string::npos)
            {
                after.push_back(content.substr(i));
                break;
            }
            after.push_back(content.substr(i, j - i));
            i = j + 1;
        }
        if (after.size() < 13)
        {
            return s;
        }
        const std::uint64_t utime = std::strtoull(after[11].c_str(), nullptr, 10);
        const std::uint64_t stime = std::strtoull(after[12].c_str(), nullptr, 10);
        s.cpu_ticks = utime + stime;
    }

    {
        std::ifstream in(base + "/status");
        std::string line;
        while (std::getline(in, line))
        {
            if (line.rfind("VmRSS:", 0) == 0)
            {
                unsigned long kb = 0;
                if (std::sscanf(line.c_str(), "VmRSS: %lu", &kb) == 1)
                {
                    s.rss_mb = static_cast<double>(kb) / 1024.0;
                }
            }
            else if (line.rfind("VmHWM:", 0) == 0)
            {
                unsigned long kb = 0;
                if (std::sscanf(line.c_str(), "VmHWM: %lu", &kb) == 1)
                {
                    s.hwm_mb = static_cast<double>(kb) / 1024.0;
                }
            }
            else if (line.rfind("Threads:", 0) == 0)
            {
                int th = 0;
                if (std::sscanf(line.c_str(), "Threads: %d", &th) == 1)
                {
                    s.threads = th;
                }
            }
        }
    }

    s.ok = true;
    return s;
}

pid_t wait_for_unique_process(const Options &opt, pid_t self_pid)
{
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(opt.wait_sec));

    while (true)
    {
        const auto pids = find_pids_by_name(opt.name, self_pid);
        if (pids.size() == 1)
        {
            return pids.front();
        }
        if (pids.size() > 1)
        {
            std::fprintf(
                stderr,
                "错误: 找到 %zu 个名为 '%s' 的进程，请只保留一个:\n",
                pids.size(),
                opt.name.c_str());
            for (pid_t p : pids)
            {
                std::fprintf(stderr, "  pid=%d\n", static_cast<int>(p));
            }
            return -1;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            std::fprintf(
                stderr,
                "错误: 等待 %.1fs 未找到进程 '%s'\n",
                opt.wait_sec,
                opt.name.c_str());
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void add_sample(Agg &agg, double cpu_pct, const ProcSample &s)
{
    ++agg.samples;
    agg.cpu_sum += cpu_pct;
    if (cpu_pct > agg.cpu_max)
    {
        agg.cpu_max = cpu_pct;
    }
    agg.rss_sum += s.rss_mb;
    if (s.rss_mb > agg.rss_max)
    {
        agg.rss_max = s.rss_mb;
    }
    if (s.hwm_mb > agg.hwm_max)
    {
        agg.hwm_max = s.hwm_mb;
    }
    agg.threads_sum += static_cast<double>(s.threads);
    if (s.threads > agg.threads_max)
    {
        agg.threads_max = s.threads;
    }
}

} // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt))
    {
        print_usage(argv[0]);
        return 1;
    }

    const pid_t self_pid = ::getpid();
    const long clk_tck = ::sysconf(_SC_CLK_TCK);
    if (clk_tck <= 0)
    {
        std::fprintf(stderr, "错误: sysconf(_SC_CLK_TCK) 失败\n");
        return 1;
    }

    std::printf(
        "percept_monitor\n"
        "  name        : %s\n"
        "  interval-ms : %d\n"
        "  warmup-sec  : %.3f\n"
        "  wait-sec    : %.3f\n",
        opt.name.c_str(),
        opt.interval_ms,
        opt.warmup_sec,
        opt.wait_sec);
    if (!opt.out_path.empty())
    {
        std::printf("  out         : %s\n", opt.out_path.c_str());
    }
    std::fflush(stdout);

    const pid_t target = wait_for_unique_process(opt, self_pid);
    if (target < 0)
    {
        return 1;
    }
    std::printf("attached pid=%d\n", static_cast<int>(target));
    std::fflush(stdout);

    FILE *csv = nullptr;
    if (!opt.out_path.empty())
    {
        csv = std::fopen(opt.out_path.c_str(), "w");
        if (csv == nullptr)
        {
            std::fprintf(stderr, "错误: 无法写入 %s\n", opt.out_path.c_str());
            return 1;
        }
        std::fprintf(
            csv,
            "elapsed_sec,phase,cpu_percent,rss_mb,hwm_mb,threads\n");
    }

    ProcSample prev = read_sample(target);
    if (!prev.ok)
    {
        std::fprintf(stderr, "错误: 无法读取 /proc/%d\n", static_cast<int>(target));
        if (csv != nullptr)
        {
            std::fclose(csv);
        }
        return 1;
    }
    auto prev_wall = std::chrono::steady_clock::now();
    const auto start_wall = prev_wall;

    Agg run_agg{};
    std::uint64_t total_samples = 0;

    while (process_alive(target))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(opt.interval_ms));
        if (!process_alive(target))
        {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        const ProcSample cur = read_sample(target);
        if (!cur.ok)
        {
            break;
        }

        const double wall_dt =
            std::chrono::duration<double>(now - prev_wall).count();
        const double elapsed =
            std::chrono::duration<double>(now - start_wall).count();

        // cpu_percent：相邻两拍 Δ(utime+stime) / CLK_TCK / 墙钟间隔
        double cpu_pct = 0.0;
        if (wall_dt > 0.0 && cur.cpu_ticks >= prev.cpu_ticks)
        {
            const double cpu_sec =
                static_cast<double>(cur.cpu_ticks - prev.cpu_ticks) /
                static_cast<double>(clk_tck);
            cpu_pct = 100.0 * cpu_sec / wall_dt;
        }

        const bool in_warmup = elapsed < opt.warmup_sec;
        const char *phase = in_warmup ? "warmup" : "run";
        ++total_samples;

        if (!in_warmup)
        {
            add_sample(run_agg, cpu_pct, cur);
        }

        if (csv != nullptr)
        {
            std::fprintf(
                csv,
                "%.3f,%s,%.2f,%.2f,%.2f,%d\n",
                elapsed,
                phase,
                cpu_pct,
                cur.rss_mb,
                cur.hwm_mb,
                cur.threads);
        }

        prev = cur;
        prev_wall = now;
    }

    if (csv != nullptr)
    {
        std::fclose(csv);
        csv = nullptr;
    }

    const double window =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_wall)
            .count();

    std::printf("\n=== percept_monitor summary ===\n");
    std::printf(
        "name=%s  pid=%d  total_samples=%llu  run_samples=%llu  interval=%dms  "
        "window=%.1fs  warmup=%.1fs\n",
        opt.name.c_str(),
        static_cast<int>(target),
        static_cast<unsigned long long>(total_samples),
        static_cast<unsigned long long>(run_agg.samples),
        opt.interval_ms,
        window,
        opt.warmup_sec);

    if (run_agg.samples == 0)
    {
        std::printf("no run-phase samples (increase --duration-sec or lower --warmup-sec)\n");
        return 0;
    }

    const double n = static_cast<double>(run_agg.samples);
    std::printf(
        "cpu_avg=%.1f%%  cpu_max=%.1f%%\n"
        "rss_avg=%.2fMB  rss_max=%.2fMB  rss_hwm=%.2fMB\n"
        "threads_avg=%.1f  threads_max=%d\n",
        run_agg.cpu_sum / n,
        run_agg.cpu_max,
        run_agg.rss_sum / n,
        run_agg.rss_max,
        run_agg.hwm_max,
        run_agg.threads_sum / n,
        run_agg.threads_max);
    return 0;
}
