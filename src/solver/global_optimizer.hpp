#ifndef _GLOBAL_OPTIMIZER_HPP // header guard
#define _GLOBAL_OPTIMIZER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

#include "Eigen/Dense" // must include Eigen BEFORE dco/c++
#include "boost/numeric/interval.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "logging.hpp"
#include "objective.hpp"
#include "utils/daeo_utils.hpp"
#include "utils/sylvesters_criterion.hpp"

using std::vector;

// DATA TYPES

enum GlobalOptimizerMode {
  FIND_ALL_LOCAL_MINIMIZERS,
  FIND_ONLY_GLOBAL_MINIMIZER
};

template <typename NUMERIC_T> struct GlobalOptimizerSettings {
  GlobalOptimizerMode MODE = FIND_ALL_LOCAL_MINIMIZERS;
  std::size_t MAXITER = 100'000;
  std::size_t MAX_REFINE_ITER = 1'000;

  NUMERIC_T TOL_Y = 1.0e-8;

  bool LOGGING_ENABLED = true;
  bool RETEST_CRITICAL_POINTS = false;
};

template <typename NUMERIC_T, typename INTERVAL_T, int YDIMS>
struct GlobalOptimizerResults {

  NUMERIC_T optima_upper_bound = std::numeric_limits<NUMERIC_T>::max();

  vector<Eigen::Vector<INTERVAL_T, YDIMS>> minima_intervals;

  vector<Eigen::Vector<INTERVAL_T, YDIMS>> hessian_test_inconclusive;
};

enum OptimizerTestCode {
  CONVERGENCE_TEST_INCONCLUSIVE = 0,
  CONVERGENCE_TEST_PASS = 1,
  VALUE_TEST_FAIL = 2,
  GRADIENT_TEST_FAIL = 4,
  GRADIENT_TEST_PASS = 8,
  HESSIAN_TEST_LOCAL_MAX = 16,
  HESSIAN_MAYBE_INDEFINITE = 32,
  HESSIAN_TEST_LOCAL_MIN = 64,
  CONSTRAINT_INFEASIBLE = 128,
  CONSTRAINT_FEASIBLE = 256,
};

inline auto format_as(OptimizerTestCode evc) { return fmt::underlying(evc); }

// UTILITY FUNCTIONS

template <typename T, int DIMS, typename P = suggested_interval_policies<T>>
Eigen::Vector<boost::numeric::interval<T, P>, DIMS>
build_box(Eigen::Vector<T, DIMS> const &ll, Eigen::Vector<T, DIMS> const &ur) {
  Eigen::Vector<boost::numeric::interval<T, P>, DIMS> res(ll.rows());
  for (size_t i = 0; i < (size_t)ll.rows(); i++) {
    res(i).set(ll(i), ur(i));
  }
  return res;
};

// OPTIMIZER CLASSES

template <int XDIMS = Eigen::Dynamic, int YDIMS = Eigen::Dynamic,
          int NPARAMS = Eigen::Dynamic, typename NUMERIC_T = double,
          typename INTERVAL_POLICIES = suggested_interval_policies<double>>
class GlobalOptimizerBase {
public:
  // Interval type for computations.
  using interval_t = boost::numeric::interval<NUMERIC_T, INTERVAL_POLICIES>;
  // Result struct type, for convenience.
  using results_t = GlobalOptimizerResults<NUMERIC_T, interval_t, YDIMS>;
  // Type of the variable on which the function should be optimized

  /**
   *@brief Eigen::Vector type of system state x
   */
  using x_t = Eigen::Vector<NUMERIC_T, XDIMS>;

  /**
   * @brief Eigen::Vector type of the search arguments y
   */
  using y_t = Eigen::Vector<NUMERIC_T, YDIMS>;

  /**
   * @brief Eigen::Vector type of the parameter vector p
   */
  using params_t = Eigen::Vector<NUMERIC_T, NPARAMS>;

  /**
   * @brief Eigen::Vector type of intervals for the search arguments y
   */
  using y_interval_t = Eigen::Vector<interval_t, YDIMS>;

  /**
   * @brief Eigen::Matrix type of the Hessian of h(...) w.r.t y
   */
  using y_hessian_t = Eigen::Matrix<NUMERIC_T, YDIMS, YDIMS>;

  /**
   * @brief Eigen::Matrix type of the Hessian of h(...) w.r.t y as an interval
   * type
   */
  using y_hessian_interval_t = Eigen::Matrix<interval_t, YDIMS, YDIMS>;

  GlobalOptimizerSettings<NUMERIC_T> const settings;

  /**
   * @brief Find minima in @c y of @c h(t,x,y;p) using the set search domain.
   * @param[in] t
   * @param[in] x
   * @param[in] y0 Box to search for minima.
   * @param[in] params
   * @returns Solver results struct.
   */
  results_t find_minima_at(NUMERIC_T t, x_t const &x, y_interval_t const &y0,
                           params_t const &params) {
    std::queue<y_interval_t> workq;
    workq.push(y0);
    size_t i = 0;
    BNBOptimizerLogger logger("bnb_minimizer_log");
    auto comp_start = std::chrono::high_resolution_clock::now();
    if (settings.LOGGING_ENABLED) {
      logger.log_computation_begin(comp_start, i, workq.front());
    }
    results_t sresults;
    // we could do this with OpenMP tasks for EASY speedup.
    while (!workq.empty() && i < settings.MAXITER) {
      y_interval_t y_i(std::move(workq.front()));
      workq.pop();
      // std::queue has no push_range on my GCC
      vector<y_interval_t> new_work =
          process_interval(i, t, x, y_i, params, sresults, logger);
      // consume the vector of new intervals
      for (auto &y : new_work) {
        workq.push(std::move(y));
      }
      // if (i % 100 == 0) {
      //   fmt::println("Iteration {:d}, work q size {:d}", i, workq.size());
      // }
      i++;
    }

    // one last check for the global.
    // am I saving any work with this? TBD.
    if (settings.MODE == FIND_ONLY_GLOBAL_MINIMIZER) {
      filter_global_optimum_from_results(sresults, t, x, params);
    }

    auto comp_end = std::chrono::high_resolution_clock::now();
    if (settings.LOGGING_ENABLED) {
      logger.log_computation_end(comp_end, i, sresults.minima_intervals.size());
    }
    return sresults;
  };

  GlobalOptimizerBase(GlobalOptimizerSettings<NUMERIC_T> t_settings)
      : settings(t_settings){};

protected:
  virtual vector<y_interval_t>
  process_interval(size_t tasknum, NUMERIC_T t, x_t const &x,
                   y_interval_t const &y_i, params_t const &params,
                   results_t &results, BNBOptimizerLogger &logger) = 0;

  virtual void filter_global_optimum_from_results(results_t &results,
                                                  NUMERIC_T t, x_t const &x,
                                                  params_t const &params) = 0;

  virtual y_interval_t narrow_via_bisection(NUMERIC_T t, x_t const &x,
                                            y_interval_t const &y_in,
                                            params_t const &params,
                                            vector<bool> &dims_converged) = 0;

  /**
   * @brief Check if the width of each dimension of @c y_i is smaller than the
   * prescribed tolerance.
   */
  vector<bool> measure_convergence(y_interval_t const &y_i) {
    vector<bool> dims_converged(y_i.rows(), true);
    for (int k = 0; k < y_i.rows(); k++) {
      dims_converged[k] = (boost::numeric::width(y_i(k)) <= settings.TOL_Y ||
                           (y_i(k).lower() == 0 && y_i(k).upper() == 0));
    }
    return dims_converged;
  }

  /**
   * @brief Check that all of the dimensions in @c y_i are are smaller than the
   * user-specified tolerance.
   */
  OptimizerTestCode convergence_test(y_interval_t const &y_i) {
    if (std::all_of(y_i.begin(), y_i.end(), [TOL = (settings.TOL_Y)](auto y) {
          return (boost::numeric::width(y) <= TOL ||
                  (y.lower() == 0 && y.upper() == 0));
        })) {
      return CONVERGENCE_TEST_PASS;
    }
    return CONVERGENCE_TEST_INCONCLUSIVE;
  };

  /**
   * @brief Check if all dimensions have converged to less than the
   * user-specified tolerance.
   */
  OptimizerTestCode convergence_test(vector<bool> const &dims_converged) {
    if (std::all_of(dims_converged.begin(), dims_converged.end(),
                    [](auto v) { return v; })) {
      return CONVERGENCE_TEST_PASS;
    }
    return CONVERGENCE_TEST_INCONCLUSIVE;
  };

  /**
   *@brief Test if the interval gradient contains zero.
   */
  OptimizerTestCode gradient_test(y_interval_t const &dhdy) {
    bool grad_pass = std::all_of(dhdy.begin(), dhdy.end(), [](interval_t ival) {
      return boost::numeric::zero_in(ival);
    });

    if (grad_pass) {
      return GRADIENT_TEST_PASS;
    }
    return GRADIENT_TEST_FAIL;
  };

  // subclasses must have own hessian tests.

  /**
   * @brief Bisects the n-dimensional range @c x in each dimension that is not
   * flagged in @c dims_converged. Additionally performs a gradient check at the
   * bisection point and discards the LEFT interval if a bisection happened
   * exactly on the optimizer point.
   * @returns @c vector of n-dimensional intervals post-split
   */
  vector<y_interval_t> bisect_interval(NUMERIC_T t, x_t const &x,
                                       y_interval_t const &y, params_t const &p,
                                       vector<bool> const &dims_converged,
                                       y_interval_t const &grad_y) {
    vector<y_interval_t> res;
    res.emplace_back(y.rows());
    res[0] = y;
    for (int i = 0; i < y.rows(); i++) {
      if (dims_converged[i]) {
        continue;
      }
      size_t result_size = res.size();
      NUMERIC_T split_point = boost::numeric::median(y(i));
      for (size_t j = 0; j < result_size; j++) {
        y_interval_t splitting_plane = res[j];
        splitting_plane(i).assign(split_point, split_point);
        if (zero_in_or_absolutely_near(grad_y(i), this->settings.TOL_Y)) {
          // capture the splitting plane in an interval smaller than TOL
          NUMERIC_T cut_L = (split_point + res[j](i).lower()) / 2;
          NUMERIC_T cut_R = (split_point + res[j](i).upper()) / 2;
          res.emplace_back(y.rows());
          res.back() = res[j];
          res.back()(i).assign(cut_L, cut_R);
          // add the right side to the end of the result vector
          res.emplace_back(y.rows());
          res.back() = res[j];
          res.back()(i).assign(cut_R, y(i).upper());
          // update res[j] to be the left side
          res[j](i).assign(res[j](i).lower(), cut_L);
        } else {
          // add the right side to the end of the result vector
          res.emplace_back(y.rows());
          res.back() = res[j];
          res.back()(i).assign(split_point, y(i).upper());
          // update res[j] to be the left side
          res[j](i).assign(res[j](i).lower(), split_point);
        }
      }
    }
    return res;
  }
};

/**
 * @class UnconstrainedGlobalOptimizer
 */
template <typename FN, int XDIMS, int YDIMS, int NPARAMS,
          typename NUMERIC_T = double,
          typename INTERVAL_POLICIES = suggested_interval_policies<NUMERIC_T>>
class UnconstrainedGlobalOptimizer
    : public GlobalOptimizerBase<XDIMS, YDIMS, NPARAMS, NUMERIC_T,
                                 INTERVAL_POLICIES> {
public:
  using base_t =
      GlobalOptimizerBase<XDIMS, YDIMS, NPARAMS, NUMERIC_T, INTERVAL_POLICIES>;
  using x_t = typename base_t::x_t;
  using interval_t = typename base_t::interval_t;
  using results_t = typename base_t::results_t;
  using y_t = typename base_t::y_t;
  using y_interval_t = typename base_t::y_interval_t;
  using params_t = typename base_t::params_t;
  using y_hessian_t = typename base_t::y_hessian_t;
  using y_hessian_interval_t = typename base_t::y_hessian_interval_t;

  UnconstrainedGlobalOptimizer(FN t_objective,
                               GlobalOptimizerSettings<NUMERIC_T> t_settings)
      : base_t(t_settings), m_objective(t_objective){};

protected:
  WrappedObjective<FN> m_objective;

  vector<y_interval_t> process_interval(size_t tasknum, NUMERIC_T t,
                                        x_t const &x, y_interval_t const &y_i,
                                        params_t const &params,
                                        results_t &results,
                                        BNBOptimizerLogger &logger) override {
    using clock = std::chrono::high_resolution_clock;
    if (this->settings.LOGGING_ENABLED) {
      logger.log_task_begin(clock::now(), tasknum, y_i);
    }
    std::vector<bool> dims_converged = this->measure_convergence(y_i);
    size_t result_code = this->convergence_test(dims_converged);
    // value test
    interval_t h(m_objective.objective_value(t, x, y_i, params));
    // fails if the lower end of the interval is larger than global upper bound
    // we update the global upper bound if the upper end of the interval is less
    // than the global optimum bound we only mark failures here, and we could
    // bail at this point, if we wished.
    // TODO bail early and avoid derivative tests
    if (results.optima_upper_bound < h.lower()) {
      result_code |= VALUE_TEST_FAIL;
    } else if (h.upper() < results.optima_upper_bound) {
      results.optima_upper_bound = h.upper();
    }

    // first derivative test
    // fails if it is not possible for the gradient to be zero inside the
    // interval y
    y_interval_t dhdy(m_objective.grad_y(t, x, y_i, params));
    result_code |= this->gradient_test(dhdy);
    // return early if gradient test fails, saves determinant computations
    if (result_code & GRADIENT_TEST_FAIL) {
      if (this->settings.LOGGING_ENABLED) {
        logger.log_task_complete(clock::now(), tasknum, y_i, result_code);
      }
      return {};
    }

    y_hessian_interval_t d2hdy2(m_objective.hess_y(t, x, y_i, params));
    result_code |= hessian_test(d2hdy2);

    vector<y_interval_t> candidate_intervals;
    if (this->settings.MODE == FIND_ONLY_GLOBAL_MINIMIZER &&
        (result_code & VALUE_TEST_FAIL)) {
      // only searching for global, value test failed
      // do nothing
    } else if (result_code & (GRADIENT_TEST_FAIL | HESSIAN_TEST_LOCAL_MAX)) {
      // gradient test failed OR hessian test failed
      // do nothing
    } else if (result_code & HESSIAN_TEST_LOCAL_MIN) {
      // gradient test pass, hessian test indicates local min
      y_interval_t y_res =
          narrow_via_bisection(t, x, y_i, params, dims_converged);
      result_code |= CONVERGENCE_TEST_PASS;
      if (this->settings.MODE == FIND_ONLY_GLOBAL_MINIMIZER) {
        interval_t h_res = m_objective.objective_value(t, x, y_i, params);
        if (h_res.lower() >= results.optima_upper_bound) {
          result_code |= VALUE_TEST_FAIL;
        } else {
          results.minima_intervals.push_back(y_res);
        }
      } else {
        results.minima_intervals.push_back(y_res);
      }
    } else if (result_code & CONVERGENCE_TEST_PASS) {
      // hessian test inconclusive, BUT the interval is too small to split
      // further save in a special list to deal with later
      results.hessian_test_inconclusive.push_back(y_i);
    } else {
      // gradient test pass, hessian test inconclusive, interval can still be
      // split!
      candidate_intervals =
          this->bisect_interval(t, x, y_i, params, dims_converged, dhdy);
    }

    if (this->settings.LOGGING_ENABLED) {
      logger.log_task_complete(clock::now(), tasknum, y_i, result_code);
    }

    return candidate_intervals;
  };

  void filter_global_optimum_from_results(results_t &results, NUMERIC_T t,
                                          x_t const &x,
                                          params_t const &params) override {
    vector<y_interval_t> res;
    NUMERIC_T h_max = std::numeric_limits<NUMERIC_T>::max();
    interval_t h;
    size_t i_star = 0;
    for (size_t i = 0; i < results.minima_intervals.size(); i++) {
      h = m_objective.objective_value(t, x, results.minima_intervals[i],
                                      params);
      if (h.upper() < h_max) {
        h_max = h.upper();
        i_star = i;
      }
    }
    res.push_back(results.minima_intervals[i_star]);
    results.minima_intervals = std::move(res);
  };

  y_interval_t narrow_via_bisection(NUMERIC_T t, x_t const &x,
                                    y_interval_t const &y_i,
                                    params_t const &params,
                                    vector<bool> &dims_converged) override {
    using boost::numeric::median;
    y_interval_t y(y_i);
    y_t y_m(y.rows());
    y_t gradient_y_m(y.rows());

    size_t iter = 0;
    while (iter < this->settings.MAX_REFINE_ITER) {
      if (this->convergence_test(dims_converged)) {
        break;
      }

      y_m = y.unaryExpr([](auto ival) { return median(ival); });
      gradient_y_m = m_objective.grad_y(t, x, y_m, params);
      for (int i = 0; i < y.size(); i++) {
        if (dims_converged[i]) {
          continue;
        }
        if (gradient_y_m(i) > 0) {
          y(i) = interval_t(y(i).lower(), y_m(i));
        } else {
          y(i) = interval_t(y_m(i), y(i).upper());
        }
        dims_converged[i] = boost::numeric::width(y(i)) <= this->settings.TOL_Y;
      }
      iter++;
    }
    return y;
  }

  OptimizerTestCode hessian_test(y_hessian_interval_t const &d2hdy2) {
    if (leading_minors_positive(d2hdy2)) {
      return HESSIAN_TEST_LOCAL_MIN;
    } else if (leading_minors_alternate(d2hdy2)) {
      return HESSIAN_TEST_LOCAL_MAX;
    }
    return HESSIAN_MAYBE_INDEFINITE;
  }
};

#endif
