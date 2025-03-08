#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename T, size_t Order=0>
class ar_op // : std::function
{
public:

  ar_op() = default;

  // template <class _InputIterator>
  // T operator()(_InputIterator begin, _InputIterator end) const {return op(begin, end);};

  // template <class _InputIterator>
  virtual T op(const T* begin, const T* end) const {};

  T op_vector(std::vector<T> input) const {return op(input.data(), input.data()+input.size());};
  T op_array(std::array<T, Order> input) const {return op(input.data(), input.data()+Order);};
};


template <typename T, typename Tp, size_t Order=0>
class ar_linear_op : public ar_op<T, Order>
{

  using ar_op<T, Order>::ar_op;

public:
  std::vector<Tp> weight;
  Tp bias;
  Tp scale;

  ar_linear_op(std::vector<Tp> weight, Tp bias, Tp scale) : weight(weight), bias(bias), scale(scale) {};

  // template <class _InputIterator>
  T op(const T* begin, const T* end) const override
  {
    assert(input.size() == weight.size());
    Tp ret=0;
    size_t i=0;
    for (; begin<end; begin++)
    {
      ret += static_cast<Tp>(*begin) * weight[i] + bias;
      i++;
    }
    return static_cast<T>(ret * scale);
  };

};

template <typename T, typename Tp, size_t Order=0>
class ar_limited_scaled_add_linear_op : public ar_op<T, Order>
{

  using ar_op<T, Order>::ar_op;

public:
  std::vector<Tp> weight;
  Tp bias;
  Tp scale;
  Tp min;
  Tp max;

  ar_limited_scaled_add_linear_op(std::vector<Tp> weight, Tp bias, Tp scale, Tp min, Tp max) : weight(weight), bias(bias), scale(scale), min(min), max(max) {};

  // template <class _InputIterator>
  T op(const T* begin, const T* end) const override
  {
    assert(input.size() == weight.size());
    Tp base=static_cast<Tp>(*(begin++));
    Tp base_unscaled=std::floor(base / scale);
    Tp adder=0;
    size_t i=0;
    for (; begin<end; begin++)
    {
      adder += static_cast<Tp>(*begin) * weight[i];
      i++;
    }
    // printf("base=%f, base_unscaled=%f, adder=%f, bias=%f, scale=%f", base, base_unscaled, adder, bias, scale);
    adder += bias;
    adder = std::round(std::max(min, std::min(max, base_unscaled + adder))) - base_unscaled;
    // printf("final_adder=%f\n", adder * scale);
    return static_cast<T>(base + adder * scale);
  };

};


template <typename T>
class ar_lookup_op : public ar_op<T>
{
public:
  std::vector<T> lut;
  std::vector<size_t> shape;

  ar_lookup_op(std::vector<T> lut, std::vector<size_t> shape) : lut(lut), shape(shape) 
  {
    assert(lut.size() == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
  };

  template <class _InputIterator>
  T operator()(_InputIterator begin, _InputIterator end) const {
    return op(begin, end);
  };

  template <class _InputIterator>
  T op(_InputIterator begin, _InputIterator end) const
  {
    assert(input.size() == shape.size());
    size_t ptr=0;
    size_t i=0;
    for (; begin<end; begin++)
    {
      assert(*begin < shape[i]);
      if (i>0) ptr *= shape[i-1];
      ptr += *begin;
      i++;
    }
    return lut[ptr];
  };

};

template <typename T>
class ar_lookup_op_4d : public ar_op<T>
{
public:
  std::vector<std::vector<std::vector<std::vector<T>>>> lut;
  std::vector<size_t> shape;

  ar_lookup_op_4d(std::vector<std::vector<std::vector<std::vector<T>>>> lut, std::vector<size_t> shape) : lut(lut), shape(shape) 
  {
    assert(lut.size() == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
  };

  template <class _InputIterator>
  T operator()(_InputIterator begin, _InputIterator end) const {
    return op(begin, end);
  };

  template <class _InputIterator>
  T op(_InputIterator begin, _InputIterator end) const
  {
    assert(input.size() == shape.size());
    size_t ptr=0;
    size_t i=0;
    for (; begin<end; begin++)
    {
      assert(*begin < shape[i]);
      if (i>0) ptr *= shape[i-1];
      ptr += *begin;
      i++;
    }
    return lut[*begin][*(begin+1)][*(begin+2)][*(begin+3)];
  };

};