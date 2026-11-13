#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash_capture {

struct CaptureOptions {
    std::string prompts_path;
    std::string output_directory;
    int max_new_tokens;
    int max_prompt_tokens;
    bool use_fp16;
};

struct PromptRecord {
    std::string id;
    std::string text;
};

struct CapturePaths {
    std::string metadata;
    std::string features;
};

struct CaptureResult {
    std::vector<llama_token> input_ids;
    std::size_t prompt_tokens = 0;
    int feature_tokens = 0;
    std::int64_t started_us = 0;
    double seconds = 0;
};

} // namespace dflash_capture
