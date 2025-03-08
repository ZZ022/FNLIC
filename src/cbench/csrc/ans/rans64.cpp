#include "rans64.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace py = pybind11;

namespace {

/* We only run this in debug mode as its costly... */
void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
                 const std::vector<int> &cdfs_sizes, int precision) {
  for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
    assert(cdfs[i][0] == 0);
    assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
    for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
      assert(cdfs[i][j + 1] > cdfs[i][j]);
    }
  }
}

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}

} // namespace

std::vector<NUMPY_ARRAY_TYPE> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
   * although it's only run once per model after training. See TF/compression
   * implementation for an optimized version. */

  std::vector<NUMPY_ARRAY_TYPE> cdf(pmf.size() + 1);
  cdf[0] = 0; /* freq 0 */

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                 [=](float p) { return std::round(p * (1 << precision)); });

  const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);

  std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                 [precision, total](NUMPY_ARRAY_TYPE p) {
                   return ((static_cast<uint64_t>(1 << precision) * p) / total);
                 });

  std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
  cdf.back() = 1 << precision;

  for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
    if (cdf[i] == cdf[i + 1]) {
      /* Try to steal frequency from low-frequency symbols */
      uint32_t best_freq = ~0u;
      int best_steal = -1;
      for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
        uint32_t freq = cdf[j + 1] - cdf[j];
        if (freq > 1 && freq < best_freq) {
          best_freq = freq;
          best_steal = j;
        }
      }

      assert(best_steal != -1);

      if (best_steal < i) {
        for (int j = best_steal + 1; j <= i; ++j) {
          cdf[j]--;
        }
      } else {
        assert(best_steal > i);
        for (int j = i + 1; j <= best_steal; ++j) {
          cdf[j]++;
        }
      }
    }
  }

  assert(cdf[0] == 0);
  assert(cdf.back() == (1 << precision));
  for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
    assert(cdf[i + 1] > cdf[i]);
  }

  return cdf;
}

void Rans64Base::init_params(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets) {
  
  if (freqs.ndim() != 2 || freqs.shape(0) != num_symbols.size()) {
    throw py::value_error("freqs should be 2-dimensional with shape (num_symbols.size(), >num_symbols.max())");
  }

  // TODO: tail_mass as parameter?
  const float tail_mass = 1.f;

  _cdfs = std::vector<std::vector<NUMPY_ARRAY_TYPE>>(freqs.shape(0));
  _cdfs_sizes = std::vector<NUMPY_ARRAY_TYPE>(freqs.shape(0));
  for (ssize_t idx=0; idx < freqs.shape(0); idx++) {
    const auto nsym = num_symbols.at(idx);
    std::vector<float> pmf(nsym + 1);
    std::vector<NUMPY_ARRAY_TYPE> freq(freqs.data(idx), freqs.data(idx) + nsym);
    const float freq_total = std::accumulate(freq.begin(), freq.end(), 0.0f) + tail_mass;
    pmf[nsym] = tail_mass / freq_total;
    std::transform(freq.begin(), freq.end(), pmf.begin(),
                  [=](NUMPY_ARRAY_TYPE f) { return static_cast<float>(f) / freq_total; });
    auto cdf = pmf_to_quantized_cdf(pmf, _freq_precision);
    _cdfs[idx] = cdf;
    _cdfs_sizes[idx] = nsym+2;
  }

  _offsets = std::vector<NUMPY_ARRAY_TYPE>(offsets.data(), offsets.data() + offsets.size());
  
  assert_cdfs(_cdfs, _cdfs_sizes, _freq_precision);

  _is_initialized = true;
}


void Rans64Base::init_cdf_params(const py::array_t<NUMPY_ARRAY_TYPE> &cdfs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &cdfs_sizes,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets) {
  
  if (cdfs.ndim() != 2 || cdfs.shape(0) != cdfs_sizes.size()) {
    throw py::value_error("cdfs should be 2-dimensional with shape (cdfs_sizes.size(), >cdfs_sizes.max())");
  }

  _cdfs = std::vector<std::vector<NUMPY_ARRAY_TYPE>>();
  for (ssize_t idx=0; idx < cdfs.shape(0); idx++) {
    _cdfs.emplace_back(cdfs.data(idx), cdfs.data(idx) + cdfs_sizes.at(idx));
  }

  _cdfs_sizes = std::vector<NUMPY_ARRAY_TYPE>(cdfs_sizes.data(), cdfs_sizes.data() + cdfs_sizes.size());
  _offsets = std::vector<NUMPY_ARRAY_TYPE>(offsets.data(), offsets.data() + offsets.size());
  
  assert_cdfs(_cdfs, _cdfs_sizes, _freq_precision);

  _is_initialized = true;

}

// void Rans64Encoder::init_params(const py::array_t<NUMPY_ARRAY_TYPE> &cdfs,
//                            const py::array_t<NUMPY_ARRAY_TYPE> &cdfs_sizes,
//                            const py::array_t<NUMPY_ARRAY_TYPE> &offsets) {
  
//   if (cdfs.ndim() != 2 || cdfs.shape(0) != cdfs_sizes.size()) {
//     throw py::value_error("cdfs should be 2-dimensional with shape (cdfs_sizes.size(), >cdfs_sizes.max())");
//   }

//   _cdfs = std::vector<std::vector<NUMPY_ARRAY_TYPE>>();
//   for (ssize_t idx=0; idx < cdfs.shape(0); idx++) {
//     _cdfs.emplace_back(cdfs.data(idx), cdfs.data(idx) + cdfs_sizes.at(idx));
//   }

//   _cdfs_sizes = std::vector<NUMPY_ARRAY_TYPE>(cdfs_sizes.data(), cdfs_sizes.data() + cdfs_sizes.size());
//   _offsets = std::vector<NUMPY_ARRAY_TYPE>(offsets.data(), offsets.data() + offsets.size());
  
//   assert_cdfs(_cdfs, _cdfs_sizes, _freq_precision);
// }

py::bytes Rans64Encoder::encode_with_indexes(
    const py::array_t<NUMPY_ARRAY_TYPE> &symbols, const py::array_t<NUMPY_ARRAY_TYPE> &indexes, 
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets, 
    const std::optional<bool> cache) {

  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  const NUMPY_ARRAY_TYPE* symbols_ptr = symbols.data();
  const NUMPY_ARRAY_TYPE* indexes_ptr = indexes.data();
  const NUMPY_ARRAY_TYPE* ar_indexes_ptr = nullptr;
  std::vector<const NUMPY_ARRAY_TYPE*> ar_offsets_ptrs;

  if (_is_ar_initialized) {
    if (!ar_offsets.has_value()) {
      throw py::value_error("ar_offsets is required for ar coding!");
    }
    if (ar_indexes.has_value()) {
      ar_indexes_ptr = ar_indexes.value().data();
    }
    // if (ar_offsets.has_value()) {
    for (ssize_t i=0; i<ar_offsets.value().shape(0); i++){
      ar_offsets_ptrs.push_back(ar_offsets.value().data(i));
    }
  }

  assert(symbols.size() == indexes.size());

  Rans64State rans;
  uint32_t *ptr = nullptr;
  std::vector<uint32_t> output;

  if (!cache.value_or(false)) {
    Rans64EncInit(&rans);

    output = std::vector<uint32_t>(indexes.size(), 0xCC); // too much space ?
    ptr = output.data() + output.size();
    assert(ptr != nullptr);

  }

  // std::vector<std::vector<ssize_t>> ar_ptr_offsets;
  // if (_is_ar_initialized) {
  //   for (auto ar_off : _ar_offsets){
  //     ar_ptr_offsets.push_back(create_ar_ptr_offsets(indexes, ar_off));
  //   }
  // }

  // backward loop on symbols from the end;
  for (ssize_t i = symbols.size()-1; i >= 0; --i) {
    NUMPY_ARRAY_TYPE cdf_idx = indexes_ptr[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < _cdfs.size());

    if (_is_ar_initialized) {
      auto ar_idx = (ar_indexes_ptr == nullptr) ? 0 : ar_indexes_ptr[i];
      // auto ar_ptr_off = ar_ptr_offsets[ar_idx];
      cdf_idx = ar_update_index(ar_offsets_ptrs, ar_idx, cdf_idx, symbols_ptr, i);
    }

    const auto &cdf = _cdfs[cdf_idx];

    const NUMPY_ARRAY_TYPE max_value = _cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    NUMPY_ARRAY_TYPE value = symbols_ptr[i] - _offsets[cdf_idx];

    uint32_t raw_val = 0;
    if (_bypass_coding) {
      if (value < 0) {
        raw_val = -2 * value - 1;
        value = max_value;
      } else if (value >= max_value) {
        raw_val = 2 * (value - max_value);
        value = max_value;
      }
    }

    assert(value >= 0);
    assert(value < _cdfs_sizes[cdf_idx] - 1);

    // std::cout << "freq: " << cdf[value + 1] - cdf[value] << std::endl;
    // std::cout << "value: " << value << std::endl;

    Rans64Symbol sym = {static_cast<uint16_t>(cdf[value]),
                  static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                  false};

    if (_bypass_coding) {

      /* Bypass coding mode (value == max_value -> sentinel flag) */
      if (value == max_value) {
        std::vector<Rans64Symbol> bypass_syms;
        /* Determine the number of bypasses (in _bypass_precision size) needed to
        * encode the raw value. */
        int32_t n_bypass = 0;
        while ((raw_val >> (n_bypass * _bypass_precision)) != 0) {
          ++n_bypass;
        }

        /* Encode number of bypasses */
        int32_t val = n_bypass;
        while (val >= _max_bypass_val) {
          bypass_syms.push_back({_max_bypass_val, 0, true});
          val -= _max_bypass_val;
        }
        bypass_syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

        /* Encode raw value */
        for (int32_t j = 0; j < n_bypass; ++j) {
          const int32_t val =
              (raw_val >> (j * _bypass_precision)) & _max_bypass_val;
          bypass_syms.push_back(
              {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
        }

        // bypass_syms should be encoded in reverse order!
        if (!cache.value_or(false)) {
          while (!bypass_syms.empty()) {
            const Rans64Symbol sym = bypass_syms.back();
            Rans64EncPutBits(&rans, &ptr, sym.start, _bypass_precision);
            bypass_syms.pop_back();
          }
        }
        else {
          _syms.insert(_syms.end(), bypass_syms.rbegin(), bypass_syms.rend());
        }

      }
    }

    if (!cache.value_or(false)) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, _freq_precision);
    }
    else {
      _syms.push_back(sym);
    }


  }

  if (!cache.value_or(false)) {
    Rans64EncFlush(&rans, &ptr);

    const int nbytes =
        std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
    return std::string(reinterpret_cast<char *>(ptr), nbytes);
  }
  else {
    // return empty string if cached
    return "";
  }

}

py::bytes Rans64Encoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  for (auto sym : _syms) {
    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, _freq_precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, _bypass_precision);
    }
  }

  Rans64EncFlush(&rans, &ptr);

  _syms.clear();

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}


py::array_t<NUMPY_ARRAY_TYPE>
Rans64Decoder::decode_with_indexes(const std::string &encoded,
                                 const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                                 const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                                 const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) {

  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  const NUMPY_ARRAY_TYPE* indexes_ptr = indexes.data();
  const NUMPY_ARRAY_TYPE* ar_indexes_ptr = nullptr;
  std::vector<const NUMPY_ARRAY_TYPE*> ar_offsets_ptrs;
  if (ar_indexes.has_value()) {
    ar_indexes_ptr = ar_indexes.value().data();
  }

  if (_is_ar_initialized) {
    if (!ar_offsets.has_value()) {
      throw py::value_error("ar_offsets is required for ar coding!");
    }
    if (ar_indexes.has_value()) {
      ar_indexes_ptr = ar_indexes.value().data();
    }
    // if (ar_offsets.has_value()) {
    for (ssize_t i=0; i<ar_offsets.value().shape(0); i++){
      ar_offsets_ptrs.push_back(ar_offsets.value().data(i));
    }
  }

  py::array_t<NUMPY_ARRAY_TYPE> output(indexes.request(true));
  NUMPY_ARRAY_TYPE* output_ptr = output.mutable_data();

  // std::vector<std::vector<ssize_t>> ar_ptr_offsets;
  // if (_is_ar_initialized) {
  //   for (auto ar_off : _ar_offsets){
  //     ar_ptr_offsets.push_back(create_ar_ptr_offsets(indexes, ar_off));
  //   }
  // }

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (ssize_t i = 0; i < indexes.size(); ++i) {
    NUMPY_ARRAY_TYPE cdf_idx = indexes_ptr[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < _cdfs.size());

    if (_is_ar_initialized) {
      auto ar_idx = (ar_indexes_ptr == nullptr) ? 0 : ar_indexes_ptr[i];
      // auto ar_ptr_off = ar_ptr_offsets[ar_idx];
      cdf_idx = ar_update_index(ar_offsets_ptrs, ar_idx, cdf_idx, output_ptr, i);
    }


    const auto &cdf = _cdfs[cdf_idx];

    const NUMPY_ARRAY_TYPE max_value = _cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const NUMPY_ARRAY_TYPE offset = _offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&rans, _freq_precision);

    const auto cdf_end = cdf.begin() + _cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], _freq_precision);

    NUMPY_ARRAY_TYPE value = static_cast<NUMPY_ARRAY_TYPE>(s);

    if (_bypass_coding) {

      if (value == max_value) {
        /* Bypass decoding mode */
        uint32_t val = Rans64DecGetBits(&rans, &ptr, _bypass_precision);
        uint32_t n_bypass = val;

        while (val == _max_bypass_val) {
          val = Rans64DecGetBits(&rans, &ptr, _bypass_precision);
          n_bypass += val;
        }

        uint32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
          val = Rans64DecGetBits(&rans, &ptr, _bypass_precision);
          assert(val <= _max_bypass_val);
          raw_val |= val << (j * _bypass_precision);
        }
        value = raw_val >> 1;
        if (raw_val & 1) {
          value = -value - 1;
        } else {
          value += max_value;
        }
      }

    }

    output_ptr[i] = value + offset;

  }

  return output;
}

py::array_t<NUMPY_ARRAY_TYPE>
Rans64Decoder::decode_stream(const py::array_t<NUMPY_ARRAY_TYPE> &indexes, 
  const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
  const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) {

  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  const NUMPY_ARRAY_TYPE* indexes_ptr = indexes.data();
  const NUMPY_ARRAY_TYPE* ar_indexes_ptr = nullptr;
  std::vector<const NUMPY_ARRAY_TYPE*> ar_offsets_ptrs;
  if (ar_indexes.has_value()) {
    ar_indexes_ptr = ar_indexes.value().data();
  }

  if (_is_ar_initialized) {
    if (!ar_offsets.has_value()) {
      throw py::value_error("ar_offsets is required for ar coding!");
    }
    if (ar_indexes.has_value()) {
      ar_indexes_ptr = ar_indexes.value().data();
    }
    // if (ar_offsets.has_value()) {
    for (ssize_t i=0; i<ar_offsets.value().shape(0); i++){
      ar_offsets_ptrs.push_back(ar_offsets.value().data(i));
    }
  }
  // TODO: stream mode autoregressive?

  py::array_t<NUMPY_ARRAY_TYPE> output(indexes.request(true));
  NUMPY_ARRAY_TYPE* output_ptr = output.mutable_data();

  assert(_ptr != nullptr);

  for (ssize_t i = 0; i < indexes.size(); ++i) {
    const NUMPY_ARRAY_TYPE cdf_idx = indexes_ptr[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < _cdfs.size());

    const auto &cdf = _cdfs[cdf_idx];

    const NUMPY_ARRAY_TYPE max_value = _cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const NUMPY_ARRAY_TYPE offset = _offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&_rans, _freq_precision);

    const auto cdf_end = cdf.begin() + _cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], _freq_precision);

    NUMPY_ARRAY_TYPE value = static_cast<NUMPY_ARRAY_TYPE>(s);

    if (_bypass_coding) {

      if (value == max_value) {
        /* Bypass decoding mode */
        uint32_t val = Rans64DecGetBits(&_rans, &_ptr, _bypass_precision);
        uint32_t n_bypass = val;

        while (val == _max_bypass_val) {
          val = Rans64DecGetBits(&_rans, &_ptr, _bypass_precision);
          n_bypass += val;
        }

        uint32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
          val = Rans64DecGetBits(&_rans, &_ptr, _bypass_precision);
          assert(val <= _max_bypass_val);
          raw_val |= val << (j * _bypass_precision);
        }
        value = raw_val >> 1;
        if (raw_val & 1) {
          value = -value - 1;
        } else {
          value += max_value;
        }
      }

    }

    output_ptr[i] = value + offset;

  }

  return output;
}

