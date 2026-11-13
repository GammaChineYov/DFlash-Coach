// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 DFlash Coach contributors
#include "capture_io.h"
#include "constants/recipe.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace dflash_capture {
namespace {

int environment_integer(const char *name, int fallback) {
    const auto *value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

const char *feature_dtype(const CaptureOptions &options) {
    return options.use_fp16 ? "float16" : "float32";
}

} // namespace

CaptureOptions read_capture_options() {
    return {
        std::getenv("DFLASH_PROMPTS"),
        std::getenv("DFLASH_FEATURES"),
        environment_integer("DFLASH_MAX_NEW", recipe::default_max_new_tokens),
        environment_integer("DFLASH_MAX_PROMPT", recipe::default_max_prompt_tokens),
        std::getenv("DFLASH_F16") != nullptr,
    };
}

CapturePaths capture_paths(const CaptureOptions &options, const std::string &id) {
    const auto prefix = options.output_directory + "/" + id;
    const auto suffix = options.use_fp16 ? ".f16" : ".f32";
    return {prefix + ".json", prefix + suffix};
}

bool reusable_capture(
    const std::string &metadata_path,
    const PromptRecord &prompt,
    const CaptureOptions &options
) {
    std::ifstream existing(metadata_path);
    if (!existing.good()) {
        return false;
    }

    const auto saved = json::parse(existing);
    if (saved.value("prompt", "") != prompt.text) {
        return false;
    }
    const auto saved_limit = saved.value(
        "max_new_tokens", recipe::default_max_new_tokens
    );
    if (saved_limit != options.max_new_tokens) {
        return false;
    }
    return saved.value("feature_dtype", "float32") == feature_dtype(options);
}

bool save_metadata(
    const std::string &metadata_path,
    const json &row,
    const CaptureOptions &options,
    CaptureResult &result
) {
    const auto temporary_path = metadata_path + ".tmp";
    std::ofstream output(temporary_path);
    json metadata = row;
    metadata["input_ids"] = result.input_ids;
    metadata["prompt_n"] = result.prompt_tokens;
    metadata["feature_shape"] = {
        result.feature_tokens,
        recipe::target_layers.size(),
        recipe::hidden_size,
    };
    metadata["feature_dtype"] = feature_dtype(options);
    metadata["max_new_tokens"] = options.max_new_tokens;
    result.seconds = (ggml_time_us() - result.started_us) / 1e6;
    metadata["seconds"] = result.seconds;
    output << metadata.dump(2);
    output.close();

    return output && std::rename(
        temporary_path.c_str(), metadata_path.c_str()
    ) == 0;
}

} // namespace dflash_capture
