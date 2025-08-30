#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#define MKL_INT size_t
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::vector;
using std::array;
using std::unordered_map;
using std::unordered_set;
using std::pair;
using std::tuple;
using std::make_pair;
using std::make_tuple;
using std::initializer_list;
using std::cin;
using std::cout;
using std::cerr;
using std::clog;
using std::endl;
using std::istream;
using std::ostream;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::string;
using std::stoi;
using std::stol;
using std::stod;
using std::sort;
using std::find;
using std::for_each;
using std::count;
using std::accumulate;
using std::copy;
using std::reverse;
using std::transform;
using std::max;
using std::min;
using std::swap;
using std::function;
using std::chrono::system_clock;
using std::chrono::steady_clock;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;
using std::chrono::time_point;
using std::size_t;
using std::nullptr_t;
using std::move;
using std::forward;
using std::enable_if;
using std::is_same;
using std::abs;
using std::fabs;
using std::sqrt;
using std::pow;
using std::exp;
using std::log;
using std::log10;
using std::sin;
using std::cos;
using std::tan;
using std::asin;
using std::acos;
using std::atan;
using std::atan2;
using std::ceil;
using std::floor;
using std::round;
using std::hash;
using std::multiplies;
using std::accumulate;
using std::inner_product;

template<size_t N> ostream & operator<<(ostream & os, array<size_t,N> const & arr)
{
    os << "[";
    for(size_t i=0;i<N-1;++i)
    {
        os << arr[i] << ",";
    }
    os << arr[N-1] << "]";
    return os;
}

struct vector_hash
{
    uint64_t operator()(vector<size_t> const & my_vector) const
    {
        hash<size_t> hasher;
        uint64_t answer = 0;
        for (size_t i : my_vector)
        {
            answer ^= hasher(i) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
        }
        return answer;
    }
};

template<size_t N> struct array_hash
{
    uint64_t operator()(array<size_t,N> const & my_arr) const
    {
        hash<size_t> hasher;
        uint64_t answer = 0;
        for (size_t i : my_arr)
        {
            answer ^= hasher(i) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
        }
        return answer;
    }
};