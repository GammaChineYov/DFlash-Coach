// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 DFlash Coach contributors
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstdint>
#include <vector>

using json = nlohmann::ordered_json;

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) return 1;
    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    auto * vocab = llama_model_get_vocab(init->model());
    if (!ctx) return 2;
    const int layers[] = {1,6,11,16,22,27,32,37};
    for (int layer : layers) llama_set_embeddings_layer_inp(ctx, layer, true);
    std::ifstream input(std::getenv("DFLASH_PROMPTS"));
    const std::string outdir = std::getenv("DFLASH_FEATURES");
    const int max_new = std::getenv("DFLASH_MAX_NEW") ? std::atoi(std::getenv("DFLASH_MAX_NEW")) : 160;
    const int max_prompt = std::getenv("DFLASH_MAX_PROMPT") ? std::atoi(std::getenv("DFLASH_MAX_PROMPT")) : 512;
    const bool half = std::getenv("DFLASH_F16") != nullptr;
    const std::string suffix = half ? ".f16" : ".f32";
    int captured = 0;
    std::string line;
    while (std::getline(input, line)) {
        const auto row = json::parse(line);
        const std::string id = row.at("id");
        const std::string text = row.at("prompt");
        const std::string meta_path = outdir + "/" + id + ".json";
        {
            std::ifstream existing(meta_path);
            if (existing.good()) {
                auto saved = json::parse(existing);
                if (saved.value("prompt", "") == text && saved.value("max_new_tokens", 160) == max_new && saved.value("feature_dtype", "float32") == (half ? "float16" : "float32")) {
                    std::printf("SKIPPED %s\n", id.c_str());
                    std::fflush(stdout);
                    continue;
                }
            }
        }
        auto ids = common_tokenize(ctx, text, false, true);
        const size_t prompt_n = ids.size();
        if (ids.empty() || ids.size() > size_t(max_prompt) || ids.size() + max_new > llama_n_ctx(ctx)) {
            std::fprintf(stderr, "Prompt length invalid for %s: %zu\n", id.c_str(), ids.size());
            return 3;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        const std::string feature_path = outdir + "/" + id + suffix;
        std::ofstream features(feature_path + ".tmp", std::ios::binary);
        if (!features) return 6;
        int pos = 0;
        const auto begin = ggml_time_us();
        for (int step = 0; step <= max_new; ++step) {
            const int count = step == 0 ? (int)prompt_n : 1;
            for (int offset = 0; offset < count; offset += 256) {
                const int chunk = std::min(256, count-offset);
                llama_batch batch = llama_batch_init(chunk, 0, 1);
                for (int j = 0; j < chunk; ++j) common_batch_add(batch, ids[pos+offset+j], pos+offset+j, {0}, offset+j==count-1);
                const int rc = llama_decode(ctx, batch);
                llama_batch_free(batch);
                if (rc) return 4;
                llama_synchronize(ctx);
                for (int j = 0; j < chunk; ++j) {
                    for (int layer : layers) {
                        auto * h = llama_get_embeddings_layer_inp(ctx, layer);
                        if (!h) return 5;
                        if (half) {
                            ggml_fp16_t values[2048];
                            ggml_fp32_to_fp16_row(h + j*2048, values, 2048);
                            features.write(reinterpret_cast<char *>(values), sizeof(values));
                        } else {
                            features.write(reinterpret_cast<char *>(h + j*2048), 2048*sizeof(float));
                        }
                    }
                }
            }
            pos += count;
            if (step == max_new) break;
            const auto * logits = llama_get_logits_ith(ctx, -1);
            const int token = int(std::max_element(logits, logits+llama_vocab_n_tokens(vocab))-logits);
            if (llama_vocab_is_eog(vocab, token)) break;
            ids.push_back(token);
        }
        features.close();
        if (!features || std::rename((feature_path+".tmp").c_str(), feature_path.c_str()) != 0) return 7;
        std::ofstream meta(meta_path + ".tmp");
        json result = row;
        result["input_ids"] = ids;
        result["prompt_n"] = prompt_n;
        result["feature_shape"] = {pos,8,2048};
        result["feature_dtype"] = half ? "float16" : "float32";
        result["max_new_tokens"] = max_new;
        result["seconds"] = (ggml_time_us()-begin)/1e6;
        meta << result.dump(2);
        meta.close();
        if (!meta || std::rename((meta_path+".tmp").c_str(), meta_path.c_str()) != 0) return 8;
        std::printf("CAPTURED %s %d tokens %.2fs total=%d\n", id.c_str(),pos,result["seconds"].get<double>(), ++captured);
        std::fflush(stdout);
    }
    return 0;
}
