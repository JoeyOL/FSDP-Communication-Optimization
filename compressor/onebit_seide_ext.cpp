// Lightweight C++ implementation of 1-bit Seide helpers using ATen.
// It mirrors the Python implementations in comm_hooks.py but runs in C++,
// working on both CPU 和 CUDA tensors via ATen.

#include <torch/extension.h>

using torch::Tensor;

// Pack signs (1 = non-negative) into uint8 (8 bits per byte, LSB first)
Tensor pack_signs_to_bytes_cpp(const Tensor& signs_bool) {
  // signs_bool: bool or uint8 tensor of shape (N,)
  auto flat = signs_bool.to(torch::kUInt8).view({-1});
  auto device = flat.device();
  auto n = flat.numel();
  auto n_pad = (8 - (n % 8)) % 8;
  if (n_pad > 0) {
    auto pad = torch::zeros({n_pad}, torch::dtype(torch::kUInt8).device(device));
    flat = torch::cat({flat, pad}, 0);
  }
  auto powers = torch::tensor({1, 2, 4, 8, 16, 32, 64, 128},
                              torch::dtype(torch::kUInt8).device(device));
  auto view = flat.view({-1, 8}); // (n_bytes, 8)
  auto mul = (view * powers);     // uint8 -> promotes internally
  auto packed = mul.sum(1).to(torch::kUInt8); // (n_bytes,)
  return packed;
}

// For each column: compute a = mean(positive), b = mean(negative), and packed signs.
std::tuple<Tensor, Tensor, Tensor> onebit_seide_per_column_cpp(const Tensor& g, int64_t col_size) {
  TORCH_CHECK(g.dim() == 1, "g must be 1D");
  auto numel = g.numel();
  TORCH_CHECK(col_size > 0, "col_size must be > 0");
  int64_t num_cols = (numel + col_size - 1) / col_size;

  auto device = g.device();
  auto dtype = g.dtype();

  auto signs = g.sign(); // -1, 0, +1
  auto signs_pos = (signs >= 0).to(dtype);
  auto signs_neg = 1.0 - signs_pos;

  int64_t n_pad = (col_size - numel % col_size) % col_size;
  Tensor g_padded, signs_pos_padded, signs_neg_padded;
  if (n_pad > 0) {
    auto zeros_g = torch::zeros({n_pad}, torch::dtype(dtype).device(device));
    g_padded = torch::cat({g, zeros_g}, 0);

    auto zeros_pos = torch::zeros({n_pad}, torch::dtype(dtype).device(device));
    auto zeros_neg = torch::zeros({n_pad}, torch::dtype(dtype).device(device));
    signs_pos_padded = torch::cat({signs_pos, zeros_pos}, 0);
    signs_neg_padded = torch::cat({signs_neg, zeros_neg}, 0);
  } else {
    g_padded = g;
    signs_pos_padded = signs_pos;
    signs_neg_padded = signs_neg;
  }

  auto g_cols = g_padded.view({num_cols, col_size});
  auto pos_mask = signs_pos_padded.view({num_cols, col_size});
  auto neg_mask = signs_neg_padded.view({num_cols, col_size});

  auto pos_sum = (g_cols * pos_mask).sum(1); // (num_cols,)
  auto pos_count = pos_mask.sum(1);          // (num_cols,)
  auto neg_sum = (g_cols * neg_mask).sum(1);
  auto neg_count = neg_mask.sum(1);

  auto zeros_like = torch::zeros_like(pos_sum);
  auto pos_count_safe = pos_count.clamp_min(1e-8);
  auto neg_count_safe = neg_count.clamp_min(1e-8);

  auto a_vec = torch::where(pos_count > 0, pos_sum / pos_count_safe, zeros_like);
  auto b_vec = torch::where(neg_count > 0, neg_sum / neg_count_safe, zeros_like);

  auto signs_bool = (signs >= 0);
  auto packed = pack_signs_to_bytes_cpp(signs_bool);

  return std::make_tuple(packed, a_vec, b_vec);
}

// Vectorized unpack signs for all ranks at once.
Tensor unpack_signs_from_bytes_all_cpp(const Tensor& packed_all, int64_t numel) {
  TORCH_CHECK(packed_all.dim() == 2, "packed_all must be (world_size, packed_bytes)");
  auto world_size = packed_all.size(0);
  auto n_bytes = packed_all.size(1);
  auto device = packed_all.device();

  int64_t n_bits = std::min<int64_t>(numel, n_bytes * 8);
  if (n_bits == 0) {
    return torch::zeros({world_size, numel},
                        torch::dtype(torch::kFloat32).device(device));
  }

  auto packed_int = packed_all.to(torch::kInt32); // (world_size, n_bytes)
  auto bit_pos = torch::arange(8, torch::dtype(torch::kInt32).device(device)); // (8,)

  auto tmp = packed_int.unsqueeze(-1);                 // (world_size, n_bytes, 1)
  auto bit_pos_b = bit_pos.unsqueeze(0).unsqueeze(0);  // (1,1,8)
  // 使用 ATen 提供的按位右移/与运算，而不是 Tensor 上的 operator>>
  auto shifted = at::bitwise_right_shift(tmp, bit_pos_b);          // (world_size, n_bytes, 8)
  auto bits_expanded = at::bitwise_and(shifted, 1);                 // (world_size, n_bytes, 8)

  auto bits_flat = bits_expanded.view({world_size, -1}).index(
      {torch::indexing::Slice(), torch::indexing::Slice(0, n_bits)}); // (world_size, n_bits)

  auto out = torch::zeros({world_size, numel},
                          torch::dtype(torch::kFloat32).device(device));
  out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, n_bits)},
                 torch::where(bits_flat == 1,
                              torch::ones_like(bits_flat, torch::dtype(torch::kFloat32)),
                              -torch::ones_like(bits_flat, torch::dtype(torch::kFloat32))));
  return out;
}

// full_sum[i] = sum_r (a_r[c] if sign_r[i] >= 0 else b_r[c])
Tensor onebit_seide_reconstruct_from_gathered_cpp(
    const Tensor& signs_all, // (world_size, packed_bytes) uint8
    const Tensor& a_all,     // (world_size, num_cols)
    const Tensor& b_all,     // (world_size, num_cols)
    int64_t numel,
    int64_t col_size) {

  TORCH_CHECK(a_all.dim() == 2 && b_all.sizes() == a_all.sizes(),
              "a_all, b_all must both be (world_size, num_cols)");
  auto world_size = a_all.size(0);
  auto num_cols = a_all.size(1);

  auto device = a_all.device();

  auto col_idx = torch::arange(numel, torch::dtype(torch::kLong).device(device)) / col_size;
  col_idx = col_idx.clamp_max(num_cols - 1);

  // (world_size, numel) signs in {+1,-1}
  auto signs_unpacked = unpack_signs_from_bytes_all_cpp(signs_all, numel);

  auto col_idx_expanded = col_idx.unsqueeze(0).expand({world_size, -1});
  col_idx_expanded = col_idx_expanded.clamp_min(0).clamp_max(num_cols - 1);
  // 确保 gather 的 index 为 int64（torch.long）
  col_idx_expanded = col_idx_expanded.to(torch::kLong);

  auto a_expanded = torch::gather(a_all, 1, col_idx_expanded);
  auto b_expanded = torch::gather(b_all, 1, col_idx_expanded);

  auto recon_all = torch::where(signs_unpacked >= 0, a_expanded, b_expanded); // (world_size, numel)
  auto full_sum = recon_all.sum(0); // (numel,)
  return full_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("onebit_seide_per_column", &onebit_seide_per_column_cpp,
        "1-bit Seide per-column stats + packed signs (C++)");
  m.def("onebit_seide_reconstruct_from_gathered", &onebit_seide_reconstruct_from_gathered_cpp,
        "1-bit Seide reconstruct from gathered signs and (a,b) (C++)");
}

