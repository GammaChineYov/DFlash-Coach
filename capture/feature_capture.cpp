// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 DFlash Coach contributors
#include "feature_capture.h"
#include "common.h"
#include "constants/recipe.h"
#include "llama-ext.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>

namespace dflash_capture {
namespace {

bool write_feature_chunk(
    llama_context *context,
    std::ofstream &output,
    int token_count,
    bool use_fp16
) {
    for (int token = 0; token < token_count; ++token) {
        for (int layer : recipe::target_layers) {
            const auto *embeddings = llama_get_embeddings_layer_inp(context, layer);
            if (!embeddings) {
                return false;
            }

            const auto *values = embeddings + token * recipe::hidden_size;
            if (use_fp16) {
                std::array<ggml_fp16_t, recipe::hidden_size> half_values;
                ggml_fp32_to_fp16_row(values, half_values.data(), recipe::hidden_size);
                output.write(
                    reinterpret_cast<const char *>(half_values.data()),
                    sizeof(half_values)
                );
            } else {
                output.write(
                    reinterpret_cast<const char *>(values),
                    recipe::hidden_size * sizeof(float)
                );
            }
        }
    }
    return true;
}

int decode_positions(
    llama_context *context,
    const CaptureResult &result,
    int count,
    std::ofstream &output,
    bool use_fp16
) {
    for (int offset = 0; offset < count; offset += recipe::decode_batch_size) {
        const int chunk = std::min(recipe::decode_batch_size, count - offset);
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        for (int token = 0; token < chunk; ++token) {
            const int position = result.feature_tokens + offset + token;
            const bool needs_logits = offset + token == count - 1;
            common_batch_add(batch, result.input_ids[position], position, {0}, needs_logits);
        }

        const int status = llama_decode(context, batch);
        llama_batch_free(batch);
        if (status) {
            return 4;
        }
        llama_synchronize(context);
        if (!write_feature_chunk(context, output, chunk, use_fp16)) {
            return 5;
        }
    }
    return 0;
}

llama_token greedy_token(llama_context *context, const llama_vocab *vocabulary) {
    const auto *logits = llama_get_logits_ith(context, -1);
    const auto *end = logits + llama_vocab_n_tokens(vocabulary);
    return static_cast<llama_token>(std::max_element(logits, end) - logits);
}

int generate_features(
    llama_context *context,
    const llama_vocab *vocabulary,
    const CaptureOptions &options,
    std::ofstream &output,
    CaptureResult &result
) {
    for (int step = 0; step <= options.max_new_tokens; ++step) {
        const int count = step == 0 ? static_cast<int>(result.prompt_tokens) : 1;
        const int status = decode_positions(context, result, count, output, options.use_fp16);
        if (status) {
            return status;
        }
        result.feature_tokens += count;
        if (step == options.max_new_tokens) {
            break;
        }

        const auto token = greedy_token(context, vocabulary);
        if (llama_vocab_is_eog(vocabulary, token)) {
            break;
        }
        result.input_ids.push_back(token);
    }
    return 0;
}

bool valid_prompt(
    llama_context *context,
    const CaptureOptions &options,
    const CaptureResult &result
) {
    if (result.input_ids.empty()) {
        return false;
    }
    if (result.prompt_tokens > static_cast<std::size_t>(options.max_prompt_tokens)) {
        return false;
    }
    return result.prompt_tokens + options.max_new_tokens <= llama_n_ctx(context);
}

} // namespace

void enable_feature_layers(llama_context *context) {
    for (int layer : recipe::target_layers) {
        llama_set_embeddings_layer_inp(context, layer, true);
    }
}

int capture_features(
    llama_context *context,
    const llama_vocab *vocabulary,
    const PromptRecord &prompt,
    const CaptureOptions &options,
    const std::string &feature_path,
    CaptureResult &result
) {
    result.input_ids = common_tokenize(context, prompt.text, false, true);
    result.prompt_tokens = result.input_ids.size();
    if (!valid_prompt(context, options, result)) {
        std::fprintf(
            stderr, "Prompt length invalid for %s: %zu\n",
            prompt.id.c_str(), result.prompt_tokens
        );
        return 3;
    }

    llama_memory_clear(llama_get_memory(context), true);
    const auto temporary_path = feature_path + ".tmp";
    std::ofstream output(temporary_path, std::ios::binary);
    if (!output) {
        return 6;
    }
    result.started_us = ggml_time_us();
    const int status = generate_features(context, vocabulary, options, output, result);
    if (status) {
        return status;
    }

    output.close();
    if (!output || std::rename(temporary_path.c_str(), feature_path.c_str()) != 0) {
        return 7;
    }
    return 0;
}

} // namespace dflash_capture
