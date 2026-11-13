#pragma once

#include <array>

namespace recipe {
inline constexpr std::array<int, 8> target_layers{1, 6, 11, 16, 22, 27, 32, 37};
inline constexpr int hidden_size = 2048;
inline constexpr int decode_batch_size = 256;
inline constexpr int default_max_new_tokens = 160;
inline constexpr int default_max_prompt_tokens = 512;
} // namespace recipe
