#pragma once

#include "types/capture.h"
#include "types/json.h"

namespace dflash_capture {

CaptureOptions read_capture_options();
CapturePaths capture_paths(const CaptureOptions &options, const std::string &id);

bool reusable_capture(
    const std::string &metadata_path,
    const PromptRecord &prompt,
    const CaptureOptions &options
);

bool save_metadata(
    const std::string &metadata_path,
    const json &row,
    const CaptureOptions &options,
    CaptureResult &result
);

} // namespace dflash_capture
