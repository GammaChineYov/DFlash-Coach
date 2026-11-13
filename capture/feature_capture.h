#pragma once

#include "types/capture.h"

namespace dflash_capture {

void enable_feature_layers(llama_context *context);

int capture_features(
    llama_context *context,
    const llama_vocab *vocabulary,
    const PromptRecord &prompt,
    const CaptureOptions &options,
    const std::string &feature_path,
    CaptureResult &result
);

} // namespace dflash_capture
