#include "bench_run.hpp"

#include "bench_common.hpp"

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/VipLiteRuntime.hpp"

#include <cstdlib>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace percept {
namespace bench {
namespace awnn {

namespace
{

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::VipLiteRuntime;

} // namespace

bool init_bench_run_context(
    const int argc,
    char **argv,
    const ModelBenchSpec &spec,
    BenchRunContext &ctx)
{
    const fs::path base = fs::current_path();
    const std::string default_model = app_resource(spec.app_name, spec.default_model).string();
    const char *default_input_name =
        (spec.default_input != nullptr) ? spec.default_input : "dog.jpg";
    const std::string default_input = app_resource(spec.app_name, default_input_name).string();

    int loops = kDefaultLoops;
    int argi = 1;
    if (argi < argc && argv[argi][0] >= '0' && argv[argi][0] <= '9')
    {
        loops = std::atoi(argv[argi]);
        ++argi;
    }

    const char *model_arg = (argi < argc) ? argv[argi] : default_model.c_str();
    const char *input_arg = (argi + 1 < argc) ? argv[argi + 1] : default_input.c_str();

    ctx.loops = loops;
    ctx.model_path = resolve_path(model_arg, base);
    ctx.input_path = resolve_path(input_arg, base);
    ctx.pack_layout = spec.pack_layout;

    if (!fs::is_regular_file(ctx.model_path))
    {
        std::fprintf(stderr, "model not found: %s\n", ctx.model_path.c_str());
        return false;
    }
    if (!fs::is_regular_file(ctx.input_path))
    {
        std::fprintf(stderr, "input not found: %s\n", ctx.input_path.c_str());
        return false;
    }

    static VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(
            stderr,
            "VipLiteRuntime init failed (check LD_LIBRARY_PATH=$PWD/lib)\n");
        return false;
    }

    static AwnnEngine::Param params;
    params.model_path = ctx.model_path;

    static AwnnEngine engine(params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return false;
    }

    const auto &model_info = engine.getInfo();
    const auto &sizes = model_info.input_sizes.at(0);
    parse_model_input_dims(sizes, spec.input_layout, ctx.model_w, ctx.model_h, ctx.model_c);

    ctx.engine = &engine;
    return true;
}

} // namespace awnn
} // namespace bench
} // namespace percept
