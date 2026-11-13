// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 DFlash Coach contributors
#include "arg.h"
#include "capture_io.h"
#include "common.h"
#include "feature_capture.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace dflash_capture {
namespace {

int capture_record(
    llama_context *context,
    const llama_vocab *vocabulary,
    const CaptureOptions &options,
    const json &row,
    int &captured
) {
    const PromptRecord prompt{
        row.at("id").get<std::string>(),
        row.at("prompt").get<std::string>(),
    };
    const auto paths = capture_paths(options, prompt.id);
    if (reusable_capture(paths.metadata, prompt, options)) {
        std::printf("SKIPPED %s\n", prompt.id.c_str());
        std::fflush(stdout);
        return 0;
    }

    CaptureResult result;
    const int status = capture_features(
        context, vocabulary, prompt, options, paths.features, result
    );
    if (status) {
        return status;
    }
    if (!save_metadata(paths.metadata, row, options, result)) {
        return 8;
    }

    std::printf(
        "CAPTURED %s %d tokens %.2fs total=%d\n",
        prompt.id.c_str(), result.feature_tokens, result.seconds, ++captured
    );
    std::fflush(stdout);
    return 0;
}

} // namespace
} // namespace dflash_capture

int main(int argc, char **argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    llama_backend_init();
    auto initialized = common_init_from_params(params);
    auto *context = initialized->context();
    const auto *vocabulary = llama_model_get_vocab(initialized->model());
    if (!context) {
        return 2;
    }

    dflash_capture::enable_feature_layers(context);
    const auto options = dflash_capture::read_capture_options();
    std::ifstream input(options.prompts_path);
    std::string line;
    int captured = 0;
    while (std::getline(input, line)) {
        const auto row = json::parse(line);
        const int status = dflash_capture::capture_record(
            context, vocabulary, options, row, captured
        );
        if (status) {
            return status;
        }
    }
    return 0;
}
