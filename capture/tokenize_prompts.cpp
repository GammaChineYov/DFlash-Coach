#include "llama.h"
#include "types/json.h"
#include "types/tokenization.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

int tokenize_record(
    const llama_vocab *vocabulary,
    const json &row,
    int limit,
    bool include_tokens,
    TokenizationSummary &summary
) {
    const std::string prompt = row.at("prompt");
    const int token_count = -llama_tokenize(
        vocabulary, prompt.c_str(), prompt.size(), nullptr, 0, false, true
    );
    const bool valid = token_count > 0 && token_count <= limit;
    summary.max_prompt_tokens = std::max(summary.max_prompt_tokens, token_count);
    summary.invalid += !valid;
    ++summary.rows;

    json result = {
        {"id", row.at("id")},
        {"prompt_tokens", token_count},
        {"valid", valid},
    };
    if (include_tokens) {
        std::vector<llama_token> ids(token_count);
        const int written = llama_tokenize(
            vocabulary, prompt.c_str(), prompt.size(),
            ids.data(), token_count, false, true
        );
        if (written != token_count) {
            return 4;
        }
        result["tokens"] = ids;
    }
    std::printf("%s\n", result.dump().c_str());
    return 0;
}

void print_summary(const TokenizationSummary &summary) {
    const json result = {
        {"summary", true},
        {"rows", summary.rows},
        {"invalid", summary.invalid},
        {"max_prompt_tokens", summary.max_prompt_tokens},
    };
    std::printf("%s\n", result.dump().c_str());
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        return 1;
    }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{
        llama_model_load_from_file(argv[1], params), llama_model_free
    };
    if (!model) {
        return 2;
    }

    const auto *vocabulary = llama_model_get_vocab(model.get());
    const int limit = std::atoi(argv[3]);
    std::ifstream input(argv[2]);
    std::string line;
    TokenizationSummary summary;
    while (std::getline(input, line)) {
        const int status = tokenize_record(
            vocabulary, json::parse(line), limit, argc == 5, summary
        );
        if (status) {
            return status;
        }
    }
    print_summary(summary);
    return summary.invalid ? 3 : 0;
}
