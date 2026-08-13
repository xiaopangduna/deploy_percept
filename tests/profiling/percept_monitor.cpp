/**
 * 旁路资源画像：按进程名定位目标，周期性读 /proc 采样，直至目标退出。
 *
 * 用法:
 *   percept_monitor --name <proc> [--interval-ms 1000] [--warmup-sec 5]
 *                   [--wait-sec 60] [--out stats.csv]
 *
 * 进程名匹配 /proc/<pid>/cmdline 的 argv[0] basename，或 /proc/<pid>/exe basename
 *（避免 TASK_COMM 15 字符截断）。找到后锁定该 PID 直至退出。
 *
 * ---------------------------------------------------------------------------
 * 监控指标与数据源（read_stat / read_status / read_npu_sample / read_ipu_counters）
 * cpu_percent、ipu_util_percent 由相邻两拍差分得到，不在 reader 内计算。
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
 * | npu_load_percent| VIP ~30s 历史窗口占用率 | /sys/kernel/debug/viplite/core_loading |
 * |                 | %（debugfs，通常需 root）| 例: Core0: Inference Time=... ms (80%) |
 * | ipu_util_percent| Sigmastar IPU 利用率 % | /proc/mi_modules/mi_ipu/debug_hal/ipu_status |
 * |                 | （两次采样 Δbusy_us）  | 累计忙时 µs；util% = 100*Δus/(Δwall_sec*1e6) |
 * |                 |                        | 对齐 SDK ipu_utilization；仅 core0；>100 钳制 |
 *
 * 生命周期：
 *   --wait-sec 内等到唯一同名进程 → attach 后按 --interval-ms 采样 →
 *   /proc/<pid> 消失则结束并打印 summary（前 --warmup-sec 不计入 summary）。
 *
 * NPU/IPU：无权限/无节点则各自跳过，不失败。不采频率/温度/整机 CPU。
 * 板端读 VIP debugfs：建议 root SSH（见 profiling/README）。
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
    int interval_ms{1000};
    double warmup_sec{5.0};
    double wait_sec{60.0};
    std::string out_path;
};

struct StatSample
{
    bool ok{false};
    std::uint64_t cpu_ticks{0}; // utime + stime（clock ticks）
};

struct StatusSample
{
    bool ok{false};
    double rss_mb{0.0}; // VmRSS
    double hwm_mb{0.0}; // VmHWM
    int threads{0};
};

struct NpuSample
{
    bool load_ok{false};
    double load_percent{0.0};
};

struct IpuCounters
{
    bool ok{false};
    std::uint64_t busy_us{0};
};

/** 由相邻两拍 Δbusy_us 得到；首拍 / 回绕时 ok=false。 */
struct IpuSample
{
    bool ok{false};
    double util_percent{0.0};
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

    std::uint64_t npu_load_samples{0};
    double npu_load_sum{0.0};
    double npu_load_max{0.0};

    std::uint64_t ipu_util_samples{0};
    double ipu_util_sum{0.0};
    double ipu_util_max{0.0};
};

enum class NpuAccess
{
    Unknown,
    Ok,
    Unavailable, // 无目录/无权限/解析失败 → 本轮会话不再重试
};

enum class IpuAccess
{
    Unknown,
    Ok,
    Unavailable, // 无节点/无权限/读失败 → 本轮会话不再重试
};

struct RunState
{
    NpuAccess npu_access{NpuAccess::Unknown};
    IpuAccess ipu_access{IpuAccess::Unknown};
    std::uint64_t prev_ipu_busy_us{0};
    bool ipu_have_prev{false};
    bool ipu_clamp_warned{false};
    StatSample prev_stat{};
    std::chrono::steady_clock::time_point prev_wall{};
    std::chrono::steady_clock::time_point start_wall{};
    Agg run_agg{};
    std::uint64_t total_samples{0};
};

constexpr const char *kVipLiteDir = "/sys/kernel/debug/viplite";
constexpr const char *kVipCoreLoading = "/sys/kernel/debug/viplite/core_loading";
constexpr const char *kIpuStatusPath = "/proc/mi_modules/mi_ipu/debug_hal/ipu_status";

void print_usage(const char *argv0)
{
    std::fprintf(
        stderr,
        "用法:\n"
        "  %s --name <proc> [--interval-ms 1000] [--warmup-sec 5]\n"
        "     [--wait-sec 60] [--out stats.csv]\n"
        "\n"
        "按进程名旁路采样 CPU%% / RSS / threads / NPU core_loading / IPU util，直至目标退出。\n"
        "读 VIP debugfs 需要 root；IPU 读 /proc/.../ipu_status（无节点则跳过）。\n",
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

void print_config(const Options &opt)
{
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

double wall_sec_between(
    std::chrono::steady_clock::time_point a,
    std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

std::string read_file_text(const char *path)
{
    std::ifstream in(path);
    if (!in)
    {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------
// /proc/<pid>/stat → cpu ticks；cpu_percent 用墙钟差分
// ---------------------------------------------------------------------------

/**
 * /proc/<pid>/stat（man proc）：
 *   格式: pid (comm) state ppid ... utime stime ...
 *   comm 可能含空格，故从最后一个 ')' 之后切字段：
 *     after[0]=state(第3字段) … after[11]=utime … after[12]=stime
 */
bool parse_stat_cpu_ticks(const std::string &content, std::uint64_t &ticks)
{
    const auto rparen = content.rfind(')');
    if (rparen == std::string::npos || rparen + 2 >= content.size())
    {
        return false;
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
        return false;
    }
    const std::uint64_t utime = std::strtoull(after[11].c_str(), nullptr, 10);
    const std::uint64_t stime = std::strtoull(after[12].c_str(), nullptr, 10);
    ticks = utime + stime;
    return true;
}

StatSample read_stat(pid_t pid)
{
    StatSample s;
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::string content;
    std::getline(in, content);
    if (content.empty() || !parse_stat_cpu_ticks(content, s.cpu_ticks))
    {
        return s;
    }
    s.ok = true;
    return s;
}

double cpu_percent(
    std::uint64_t prev_ticks,
    std::uint64_t cur_ticks,
    double wall_dt,
    long clk_tck)
{
    if (wall_dt <= 0.0 || cur_ticks < prev_ticks || clk_tck <= 0)
    {
        return 0.0;
    }
    const double cpu_sec =
        static_cast<double>(cur_ticks - prev_ticks) / static_cast<double>(clk_tck);
    return 100.0 * cpu_sec / wall_dt;
}

// ---------------------------------------------------------------------------
// /proc/<pid>/status → VmRSS / VmHWM / Threads（瞬时值，无差分）
// ---------------------------------------------------------------------------

StatusSample read_status(pid_t pid)
{
    StatusSample s;
    std::ifstream in("/proc/" + std::to_string(pid) + "/status");
    if (!in)
    {
        return s;
    }
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
    s.ok = true;
    return s;
}

// ---------------------------------------------------------------------------
// /sys/kernel/debug/viplite/core_loading → 内核已给出的窗口占用率
// ---------------------------------------------------------------------------

/**
 * 解析 core_loading，例如:
 *   VIP Total Time=30009 ms
 *   History Loading -----> Core0: Inference Time=24063 ms (80%)
 * 取括号内百分比作为 NPU 历史窗口占用率。
 */
bool parse_core_loading(const std::string &text, double &load_percent)
{
    const auto pos = text.rfind('(');
    if (pos == std::string::npos)
    {
        return false;
    }
    int pct = -1;
    if (std::sscanf(text.c_str() + pos, "(%d%%)", &pct) != 1 || pct < 0)
    {
        return false;
    }
    load_percent = static_cast<double>(pct);
    return true;
}

NpuSample read_npu_sample(NpuAccess &access)
{
    NpuSample npu;
    if (access == NpuAccess::Unavailable)
    {
        return npu;
    }

    if (!fs::is_directory(kVipLiteDir))
    {
        if (access == NpuAccess::Unknown)
        {
            std::fprintf(
                stderr,
                "提示: 无 %s，跳过 NPU 采样（需 vipcore + debugfs）\n",
                kVipLiteDir);
            access = NpuAccess::Unavailable;
        }
        return npu;
    }

    const std::string loading = read_file_text(kVipCoreLoading);
    if (loading.empty())
    {
        if (access == NpuAccess::Unknown)
        {
            std::fprintf(
                stderr,
                "提示: 无法读 %s，跳过 NPU 采样（请以 root 运行）\n",
                kVipCoreLoading);
            access = NpuAccess::Unavailable;
        }
        return npu;
    }

    if (!parse_core_loading(loading, npu.load_percent))
    {
        if (access == NpuAccess::Unknown)
        {
            std::fprintf(stderr, "提示: core_loading 解析失败，跳过 NPU 采样\n");
            access = NpuAccess::Unavailable;
        }
        return npu;
    }

    npu.load_ok = true;
    if (access == NpuAccess::Unknown)
    {
        std::printf("npu: enabled (core_loading)\n");
        access = NpuAccess::Ok;
    }
    return npu;
}

// ---------------------------------------------------------------------------
// /proc/mi_modules/mi_ipu/debug_hal/ipu_status → 累计 busy µs；利用率差分另算
// ---------------------------------------------------------------------------

bool read_ipu_busy_us(std::uint64_t &busy_us)
{
    std::ifstream in(kIpuStatusPath);
    if (!in)
    {
        return false;
    }
    unsigned long long v = 0;
    if (!(in >> v))
    {
        return false;
    }
    busy_us = static_cast<std::uint64_t>(v);
    return true;
}

IpuCounters read_ipu_counters(IpuAccess &access)
{
    IpuCounters c;
    if (access == IpuAccess::Unavailable)
    {
        return c;
    }

    std::uint64_t busy_us = 0;
    if (!read_ipu_busy_us(busy_us))
    {
        if (access == IpuAccess::Unknown)
        {
            std::fprintf(
                stderr,
                "提示: 无法读 %s，跳过 IPU 采样（需 mi_ipu debug_hal）\n",
                kIpuStatusPath);
            access = IpuAccess::Unavailable;
        }
        return c;
    }

    if (access == IpuAccess::Unknown)
    {
        std::printf("ipu: enabled (ipu_status)\n");
        access = IpuAccess::Ok;
    }
    c.ok = true;
    c.busy_us = busy_us;
    return c;
}

/**
 * 对齐 SDK ipu_utilization：累计忙时 µs 用墙钟间隔差分得利用率。
 * 首拍（!have_prev 或 wall_dt<=0）只记录计数，不算 %。
 * 回绕：本拍 ok=false 并刷新 prev；>100% 钳制并仅首次提示。
 */
IpuSample ipu_util_from_delta(
    const IpuCounters &cur,
    std::uint64_t &prev_busy_us,
    bool &have_prev,
    double wall_dt,
    bool &clamp_warned)
{
    IpuSample ipu;
    if (!cur.ok)
    {
        return ipu;
    }

    if (!have_prev || wall_dt <= 0.0)
    {
        prev_busy_us = cur.busy_us;
        have_prev = true;
        return ipu;
    }

    if (cur.busy_us < prev_busy_us)
    {
        prev_busy_us = cur.busy_us;
        return ipu;
    }

    const double delta_us = static_cast<double>(cur.busy_us - prev_busy_us);
    double pct = 100.0 * delta_us / (wall_dt * 1e6);
    if (pct > 100.0)
    {
        if (!clamp_warned)
        {
            std::fprintf(stderr, "提示: ipu_util_percent > 100，已钳制到 100\n");
            clamp_warned = true;
        }
        pct = 100.0;
    }

    ipu.util_percent = pct;
    ipu.ok = true;
    prev_busy_us = cur.busy_us;
    return ipu;
}

void probe_accelerators(RunState &st)
{
    (void)read_npu_sample(st.npu_access);
    const IpuCounters c = read_ipu_counters(st.ipu_access);
    (void)ipu_util_from_delta(
        c, st.prev_ipu_busy_us, st.ipu_have_prev, 0.0, st.ipu_clamp_warned);
}

// ---------------------------------------------------------------------------
// 编排：聚合、CSV、一拍采样、summary
// ---------------------------------------------------------------------------

void add_sample(
    Agg &agg,
    double cpu_pct,
    const StatusSample &st,
    const NpuSample &npu,
    const IpuSample &ipu)
{
    ++agg.samples;
    agg.cpu_sum += cpu_pct;
    if (cpu_pct > agg.cpu_max)
    {
        agg.cpu_max = cpu_pct;
    }
    agg.rss_sum += st.rss_mb;
    if (st.rss_mb > agg.rss_max)
    {
        agg.rss_max = st.rss_mb;
    }
    if (st.hwm_mb > agg.hwm_max)
    {
        agg.hwm_max = st.hwm_mb;
    }
    agg.threads_sum += static_cast<double>(st.threads);
    if (st.threads > agg.threads_max)
    {
        agg.threads_max = st.threads;
    }

    if (npu.load_ok)
    {
        ++agg.npu_load_samples;
        agg.npu_load_sum += npu.load_percent;
        if (npu.load_percent > agg.npu_load_max)
        {
            agg.npu_load_max = npu.load_percent;
        }
    }

    if (ipu.ok)
    {
        ++agg.ipu_util_samples;
        agg.ipu_util_sum += ipu.util_percent;
        if (ipu.util_percent > agg.ipu_util_max)
        {
            agg.ipu_util_max = ipu.util_percent;
        }
    }
}

void csv_write_optional(FILE *csv, bool ok, double v, bool last)
{
    if (ok)
    {
        std::fprintf(csv, "%.1f", v);
    }
    std::fputc(last ? '\n' : ',', csv);
}

void write_csv_row(
    FILE *csv,
    double elapsed,
    const char *phase,
    double cpu_pct,
    const StatusSample &st,
    const NpuSample &npu,
    const IpuSample &ipu)
{
    if (csv == nullptr)
    {
        return;
    }
    std::fprintf(
        csv,
        "%.3f,%s,%.2f,%.2f,%.2f,%d,",
        elapsed,
        phase,
        cpu_pct,
        st.rss_mb,
        st.hwm_mb,
        st.threads);
    csv_write_optional(csv, npu.load_ok, npu.load_percent, false);
    csv_write_optional(csv, ipu.ok, ipu.util_percent, true);
}

bool open_csv(const std::string &path, FILE *&csv)
{
    csv = nullptr;
    if (path.empty())
    {
        return true;
    }
    csv = std::fopen(path.c_str(), "w");
    if (csv == nullptr)
    {
        std::fprintf(stderr, "错误: 无法写入 %s\n", path.c_str());
        return false;
    }
    std::fprintf(
        csv,
        "elapsed_sec,phase,cpu_percent,rss_mb,hwm_mb,threads,"
        "npu_load_percent,ipu_util_percent\n");
    return true;
}

void close_csv(FILE *&csv)
{
    if (csv != nullptr)
    {
        std::fclose(csv);
        csv = nullptr;
    }
}

bool sample_once(pid_t target, const Options &opt, long clk_tck, RunState &st, FILE *csv)
{
    const auto now = std::chrono::steady_clock::now();
    const StatSample stat = read_stat(target);
    if (!stat.ok)
    {
        return false;
    }

    const StatusSample status = read_status(target);
    const NpuSample npu = read_npu_sample(st.npu_access);

    const double wall_dt = wall_sec_between(st.prev_wall, now);
    const double elapsed = wall_sec_between(st.start_wall, now);

    const IpuCounters ipu_raw = read_ipu_counters(st.ipu_access);
    const IpuSample ipu = ipu_util_from_delta(
        ipu_raw, st.prev_ipu_busy_us, st.ipu_have_prev, wall_dt, st.ipu_clamp_warned);
    const double cpu_pct =
        cpu_percent(st.prev_stat.cpu_ticks, stat.cpu_ticks, wall_dt, clk_tck);

    const bool in_warmup = elapsed < opt.warmup_sec;
    const char *phase = in_warmup ? "warmup" : "run";
    ++st.total_samples;

    if (!in_warmup)
    {
        add_sample(st.run_agg, cpu_pct, status, npu, ipu);
    }
    write_csv_row(csv, elapsed, phase, cpu_pct, status, npu, ipu);

    st.prev_stat = stat;
    st.prev_wall = now;
    return true;
}

void run_sampling_loop(pid_t target, const Options &opt, long clk_tck, RunState &st, FILE *csv)
{
    while (process_alive(target))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(opt.interval_ms));
        if (!process_alive(target))
        {
            break;
        }
        if (!sample_once(target, opt, clk_tck, st, csv))
        {
            break;
        }
    }
}

void print_summary(const Options &opt, pid_t target, const RunState &st, double window)
{
    std::printf("\n=== percept_monitor summary ===\n");
    std::printf(
        "name=%s  pid=%d  total_samples=%llu  run_samples=%llu  interval=%dms  "
        "window=%.1fs  warmup=%.1fs\n",
        opt.name.c_str(),
        static_cast<int>(target),
        static_cast<unsigned long long>(st.total_samples),
        static_cast<unsigned long long>(st.run_agg.samples),
        opt.interval_ms,
        window,
        opt.warmup_sec);

    if (st.run_agg.samples == 0)
    {
        std::printf("no run-phase samples (run the target longer or lower --warmup-sec)\n");
        return;
    }

    const double n = static_cast<double>(st.run_agg.samples);
    std::printf(
        "cpu_avg=%.1f%%  cpu_max=%.1f%%\n"
        "rss_avg=%.2fMB  rss_max=%.2fMB  rss_hwm=%.2fMB\n"
        "threads_avg=%.1f  threads_max=%d\n",
        st.run_agg.cpu_sum / n,
        st.run_agg.cpu_max,
        st.run_agg.rss_sum / n,
        st.run_agg.rss_max,
        st.run_agg.hwm_max,
        st.run_agg.threads_sum / n,
        st.run_agg.threads_max);

    if (st.run_agg.npu_load_samples > 0)
    {
        std::printf(
            "npu_load_avg=%.1f%%  npu_load_max=%.1f%%  (samples=%llu)\n",
            st.run_agg.npu_load_sum / static_cast<double>(st.run_agg.npu_load_samples),
            st.run_agg.npu_load_max,
            static_cast<unsigned long long>(st.run_agg.npu_load_samples));
    }
    if (st.run_agg.ipu_util_samples > 0)
    {
        std::printf(
            "ipu_util_avg=%.1f%%  ipu_util_max=%.1f%%  (samples=%llu)\n",
            st.run_agg.ipu_util_sum / static_cast<double>(st.run_agg.ipu_util_samples),
            st.run_agg.ipu_util_max,
            static_cast<unsigned long long>(st.run_agg.ipu_util_samples));
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

    print_config(opt);

    const pid_t target = wait_for_unique_process(opt, self_pid);
    if (target < 0)
    {
        return 1;
    }
    std::printf("attached pid=%d\n", static_cast<int>(target));
    std::fflush(stdout);

    RunState st;
    probe_accelerators(st);

    FILE *csv = nullptr;
    if (!open_csv(opt.out_path, csv))
    {
        return 1;
    }

    st.prev_stat = read_stat(target);
    if (!st.prev_stat.ok)
    {
        std::fprintf(stderr, "错误: 无法读取 /proc/%d/stat\n", static_cast<int>(target));
        close_csv(csv);
        return 1;
    }
    st.prev_wall = std::chrono::steady_clock::now();
    st.start_wall = st.prev_wall;

    run_sampling_loop(target, opt, clk_tck, st, csv);
    close_csv(csv);

    const double window = wall_sec_between(st.start_wall, std::chrono::steady_clock::now());
    print_summary(opt, target, st, window);
    return 0;
}
