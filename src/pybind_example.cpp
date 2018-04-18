#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Stats { double mean; double std; };

auto rolling_stats(py::array_t<double> arr, size_t window) {
    if (arr.ndim() != 1)
        throw std::runtime_error("expected 1-D array");

    py::array_t<Stats> stats(arr.size());
    auto a = arr.unchecked<1>();
    auto s = stats.mutable_unchecked<1>();

    double sum = 0, sqr = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i >= window) {
            auto x = a(i - window); sum -= x; sqr -= x * x;
        }
        auto x = a(i); sum += x; sqr += x * x;
        double n = i >= window ? window : (i + 1);
        double mean = sum / n;
        s(i) = { mean, std::sqrt((sqr - sum * mean) / (n - 1)) };
    }
    return stats;
}

PYBIND11_MODULE(pybind_example, m) {
    PYBIND11_NUMPY_DTYPE(Stats, mean, std);
    m.def("rolling_stats", rolling_stats);
}
