/* Copyright (c) 2021-2022, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "rans64.h"

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {

/* We only run this in debug mode as its costly... */
void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
                 const std::vector<int> &cdfs_sizes) {
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

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    int32_t value = symbols[i] - offsets[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0);
    assert(value < cdfs_sizes[cdf_idx] - 1);

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }
    }
  }
}

void BufferedRansEncoder::encode_with_indexes_np(
    const py::array_t<int32_t> &symbols,
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets) {
  // TODO: this copys memory! Is there a way to avoid this?
  std::vector<int32_t> symbols_vec(symbols.data(), symbols.data() + symbols.size());
  std::vector<int32_t> indexes_vec(indexes.data(), indexes.data() + indexes.size());

  std::vector<std::vector<int32_t>> cdfs_vec;
  if (cdfs.ndim() != 2 && cdfs.shape(0) != cdfs_sizes.size()) {
    throw pybind11::value_error("cdfs should be 2-dimensional with shape (cdfs_sizes.size, cdfs_sizes)");
  }
  for (int32_t idx=0; idx < cdfs.shape(0); idx++){
    cdfs_vec.emplace_back(cdfs.data(idx), cdfs.data(idx) + cdfs_sizes.at(idx));
  }
  
  std::vector<int32_t> cdfs_sizes_vec(cdfs_sizes.data(), cdfs_sizes.data() + cdfs_sizes.size());
  std::vector<int32_t> offsets_vec(offsets.data(), offsets.data() + offsets.size());

  encode_with_indexes(symbols_vec, indexes_vec, cdfs_vec, cdfs_sizes_vec, offsets_vec);
}

py::bytes BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}

py::bytes
RansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets) {

  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                        offsets);
  return buffered_rans_enc.flush();
}

py::bytes
RansEncoder::encode_with_indexes_np(
    const py::array_t<int32_t> &symbols,
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets) {
  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes_np(symbols, indexes, cdfs, cdfs_sizes,
                                        offsets);
  return buffered_rans_enc.flush();
}


std::vector<int32_t>
RansDecoder::decode_with_indexes(const std::string &encoded,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

py::array_t<int32_t> RansDecoder::decode_with_indexes_np(
    const std::string &encoded,
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets) {
  // TODO: this copys memory! Is there a way to avoid this?
  std::vector<int32_t> indexes_vec(indexes.data(), indexes.data() + indexes.size());

  std::vector<std::vector<int32_t>> cdfs_vec;
  if (cdfs.ndim() != 2 && cdfs.shape(0) != cdfs_sizes.size()) {
    throw pybind11::value_error("cdfs should be 2-dimensional with shape (cdfs_sizes.size, cdfs_sizes)");
  }
  for (int32_t idx=0; idx < cdfs.shape(0); idx++){
    cdfs_vec.emplace_back(cdfs.data(idx), cdfs.data(idx) + cdfs_sizes.at(idx));
  }
  
  std::vector<int32_t> cdfs_sizes_vec(cdfs_sizes.data(), cdfs_sizes.data() + cdfs_sizes.size());
  std::vector<int32_t> offsets_vec(offsets.data(), offsets.data() + offsets.size());

  std::vector<int32_t> output = decode_with_indexes(encoded, indexes_vec, cdfs_vec, cdfs_sizes_vec, offsets_vec);
  
  // TODO: reshape output as indexes
  return py::array_t<int32_t>(output.size(), output.data());
}

void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = (uint32_t *)_stream.data();
  assert(ptr != nullptr);
  _ptr = ptr;
  Rans64DecInit(&_rans, &_ptr);
}

std::vector<int32_t>
RansDecoder::decode_stream(const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  assert(_ptr != nullptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

py::array_t<int32_t> RansDecoder::decode_stream_np(
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets) {
  // TODO: this copys memory! Is there a way to avoid this?
  std::vector<int32_t> indexes_vec(indexes.data(), indexes.data() + indexes.size());

  std::vector<std::vector<int32_t>> cdfs_vec;
  if (cdfs.ndim() != 2 && cdfs.shape(0) != cdfs_sizes.size()) {
    throw pybind11::value_error("cdfs should be 2-dimensional with shape (cdfs_sizes.size, cdfs_sizes)");
  }
  for (int32_t idx=0; idx < cdfs.shape(0); idx++){
    cdfs_vec.emplace_back(cdfs.data(idx), cdfs.data(idx) + cdfs_sizes.at(idx));
  }
  
  std::vector<int32_t> cdfs_sizes_vec(cdfs_sizes.data(), cdfs_sizes.data() + cdfs_sizes.size());
  std::vector<int32_t> offsets_vec(offsets.data(), offsets.data() + offsets.size());

  std::vector<int32_t> output = decode_stream(indexes_vec, cdfs_vec, cdfs_sizes_vec, offsets_vec);
  // TODO: reshape output as indexes
  return py::array_t<int32_t>(output.size(), output.data());
}

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
   * although it's only run once per model after training. See TF/compression
   * implementation for an optimized version. */

  for (float p : pmf) {
    if (p < 0 || !std::isfinite(p)) {
      throw std::domain_error(
          std::string("Invalid `pmf`, non-finite or negative element found: ") +
          std::to_string(p));
    }
  }

  std::vector<uint32_t> cdf(pmf.size() + 1);
  cdf[0] = 0; /* freq 0 */

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                 [=](float p) { return std::round(p * (1 << precision)); });

  const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);
  if (total == 0) {
    throw std::domain_error("Invalid `pmf`: at least one element must have a "
                            "non-zero probability.");
  }

  std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                 [precision, total](uint32_t p) {
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

py::array_t<uint32_t> pmf_to_quantized_cdf_np(
    const py::array_t<float> &pmf, int precision
    ) {
  if (pmf.ndim() == 1) {
    std::vector<float> pmf_vec(pmf.data(), pmf.data() + pmf.size());
    std::vector<uint32_t> quantized_cdf = pmf_to_quantized_cdf(pmf_vec, precision);
    return py::array_t<uint32_t>(quantized_cdf.size(), quantized_cdf.data());
  }
  // batched processing
  else {
    const ssize_t ndim = pmf.ndim();
    const ssize_t num_symbols = pmf.shape(ndim-1);
    const ssize_t batch_size = pmf.size() / num_symbols;
    // resize as batched
    py::array_t<float> batched_pmf(pmf);
    batched_pmf = batched_pmf.reshape({batch_size, num_symbols});
    py::array_t<uint32_t> out;
    out.resize({batch_size, num_symbols+1});
    for (ssize_t i=0;i<batch_size;i++){
      std::vector<float> pmf_vec(batched_pmf.data(i), batched_pmf.data(i) + num_symbols);
      std::vector<uint32_t> quantized_cdf = pmf_to_quantized_cdf(pmf_vec, precision);
      std::copy(quantized_cdf.begin(), quantized_cdf.end(), out.mutable_data(i));
    }
    return out;
  }
}



PYBIND11_MODULE(rans, m) {
  // m.attr("__name__") = "compressai.ans";

  m.doc() = "range Asymmetric Numeral System python bindings";

  py::class_<BufferedRansEncoder>(m, "BufferedRansEncoder", py::module_local())
      .def(py::init<>())
      .def("encode_with_indexes", &BufferedRansEncoder::encode_with_indexes)
      .def("encode_with_indexes_np", &BufferedRansEncoder::encode_with_indexes_np)
      .def("flush", &BufferedRansEncoder::flush);

  py::class_<RansEncoder>(m, "RansEncoder", py::module_local())
      .def(py::init<>())
      .def("encode_with_indexes", &RansEncoder::encode_with_indexes)
      .def("encode_with_indexes_np", &RansEncoder::encode_with_indexes_np);

  py::class_<RansDecoder>(m, "RansDecoder", py::module_local())
      .def(py::init<>())
      .def("set_stream", &RansDecoder::set_stream)
      .def("decode_stream", &RansDecoder::decode_stream)
      .def("decode_stream_np", &RansDecoder::decode_stream_np)
      .def("decode_with_indexes", &RansDecoder::decode_with_indexes,
           "Decode a string to a list of symbols")
      .def("decode_with_indexes_np", &RansDecoder::decode_with_indexes_np,
           "Decode a string to a list of symbols");

  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf,
        "Return quantized CDF for a given PMF", py::arg("pmf"), py::arg("precision")=precision);
  m.def("pmf_to_quantized_cdf_np", &pmf_to_quantized_cdf_np,
        "Return quantized CDF for a given PMF", py::arg("pmf"), py::arg("precision")=precision);

}
