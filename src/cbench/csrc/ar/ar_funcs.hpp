#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;


template <typename T>
T ar_3way_mean(T i1, T i2, T i3){
  return (i1+i2+i3) / 3;
}

template <typename T>
T ar_3way_mean_array(std::array<T, 3> input){
  return (input[0]+input[1]+input[2]) / 3;
}

template <typename T>
class ar_op // : std::function
{
public:

  ar_op() = default;

  template <class _InputIterator>
  T operator()(_InputIterator begin, _InputIterator end) const {};

  template <class _InputIterator>
  T op(_InputIterator begin, _InputIterator end) const;
  T op_vector(std::vector<T> input) const;
};


template <typename T>
class ar_linear_op : public ar_op<T>
{
public:
  std::vector<T> weight;
  T bias;

  ar_linear_op(std::vector<T> weight, T bias) : weight(weight), bias(bias) {};
  
  template <class _InputIterator>
  T operator()(_InputIterator begin, _InputIterator end) const {
    return op(begin, end);
  };

  template <class _InputIterator>
  T op(_InputIterator begin, _InputIterator end) const
  {
    assert(input.size() == weight.size());
    T ret=0;
    size_t i=0;
    for (; begin<end; begin++)
    {
      ret += (*begin) * weight[i] + bias;
      i++;
    }
    return ret;
  };

  T op_vector(std::vector<T> input) const
  {
    return op(input.begin(), input.end());
  };

  T op_array(std::array<T, 3> input) const
  {
    return op(input.begin(), input.end());
    // printf("op_array\n");
    // return (input[0]+input[1]+input[2]) / 3;
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
  
  T op_vector(std::vector<size_t> input) const
  {
    return op(input.begin(), input.end());
  };

  T op_array(std::array<size_t, 3> input) const
  {
    return op(input.begin(), input.end());
  };

};