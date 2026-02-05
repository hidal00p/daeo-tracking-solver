#include <cmath>

#include <fmt/core.h>

#include "Eigen/Dense"

#include "solver/global_optimizer.hpp"

template <typename T, typename XT, typename YT, int YDIMS, int PDIMS>
Eigen::Vector<std::common_type_t<XT, YT>, 2>
xprime(T t, Eigen::Vector<XT, 2> const &x, Eigen::Vector<YT, YDIMS> const &y,
       Eigen::Vector<T, PDIMS> const &p) {
  Eigen::Vector<std::common_type_t<XT, YT>, 2> res(x(1), y(0));
  return res;
}

int main(int argc, char **argv) {

  using boost::numeric::median;

  auto h =
      [](const auto t, const auto &x, const auto &y, const auto &p) -> auto{
    return (pow(x(0) - y(0), 2) + sin(p(0) * y(0)));
  };

  // auto f = [](auto t, auto const &x, auto const &y, auto const &p) {
  //   return xprime(t, x, y, p);
  // };

  GlobalOptimizerSettings<double> gopt_settings;
  gopt_settings.MODE = FIND_ALL_LOCAL_MINIMIZERS;
  gopt_settings.LOGGING_ENABLED = true;
  gopt_settings.TOL_Y = 1.0e-8;

  using gopt_t = UnconstrainedGlobalOptimizer<decltype(h), 2, 1, 1>;
  gopt_t gopt(h, gopt_settings);

  auto search_domain =
      build_box(Eigen::Vector<double, 1>(-1.0), Eigen::Vector<double, 1>(4.0));
  gopt_t::x_t x0(1.0, 1.0);
  gopt_t::params_t p(5.0);
  auto res = gopt.find_minima_at(0.0, x0, search_domain, p);

  fmt::println("at x=1.0\n");
  for (auto &y_i : res.minima_intervals) {
    fmt::println("{:: .8e}",
                 y_i.unaryExpr([](auto ival) { return median(ival); }));
  }

  gopt_t::x_t x1(3.0, 1.0);
  res = gopt.find_minima_at(0.0, x1, search_domain, p);
  fmt::println("\nat x=3\n");
  for (auto &y_i : res.minima_intervals) {
    fmt::println("{:: .8e}",
                 y_i.unaryExpr([](auto ival) { return median(ival); }));
  }
}
