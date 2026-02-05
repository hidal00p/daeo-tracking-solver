/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of solver for Differential-Algebriac Equations with
 * Embedded Optimization Criteria.
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "global_optimizer.hpp"
#include "utils/daeo_utils.hpp"
#include "xprime.hpp"

using std::vector;

enum DAEOSolverFlags {
  ONLY_USE_GLOBAL_OPTIMIZATION = 1,
  TRACK_LOCAL_OPTIMA = 2,
  DETECT_AND_CORRECT_EVENTS = 4,
  LINEARIZE_OPTIMIZER_DRIFT = 8,
};

template <typename NUMERIC_T> struct DAEOSolverSettings {
  NUMERIC_T y0_min;
  NUMERIC_T y0_max;

  size_t SEARCH_FREQUENCY = 20;
  size_t MAX_NEWTON_ITERATIONS = 120;
  NUMERIC_T NEWTON_EPS = 1.0e-8;
  NUMERIC_T EVENT_DETECTION_EPS = 5.0e-6; // this may be computeable from limits
  NUMERIC_T EVENT_DRIFT_COEFF = 0.1;      // these might always be the same.

  size_t solver_flags = TRACK_LOCAL_OPTIMA & DETECT_AND_CORRECT_EVENTS;
  bool LOGGING_ENABLED = true;
};

template <typename NUMERIC_T, int XDIMS, int YDIMS> struct DAEOSolutionState {
  NUMERIC_T t;
  Eigen::Vector<NUMERIC_T, XDIMS> x;
  size_t i_star;
  vector<Eigen::Vector<NUMERIC_T, YDIMS>> y;

  /**
   * @brief Return the number of local optima present at the this point in time.
   */
  inline size_t n_local_optima() const { return y.size(); }

  // Eigen should help with compile time optimization for this.
  // TODO make if constexpr(Eigen::Dynamic) if necessary.

  /**
   * @brief Return the number of dimensions of each local optimizer.
   */
  inline int ydims() const { return y[0].rows(); }

  /**
   * @brief Return the number of dimensions of the dynamic state @c x
   */
  inline int xdims() const { return x.rows(); }

  // what the heck am I doing
  /**
   * @brief Get a (const) reference to the global optimum y*
   */
  inline Eigen::Vector<NUMERIC_T, YDIMS> &y_star() { return y[i_star]; }
  inline const Eigen::Vector<NUMERIC_T, YDIMS> &y_star() const {
    return y[i_star];
  }
};

template <typename XPRIME, typename OBJECTIVE, typename NUMERIC_T, int XDIMS,
          int YDIMS, int NPARAMS>
class DAEOTrackingSolver {
public:
  /**
   * @brief Type of the local optimizer.
   */
  using optimizer_t =
      UnconstrainedGlobalOptimizer<OBJECTIVE, XDIMS, YDIMS, NPARAMS, NUMERIC_T>;

  /**
   * @brief Type for an optimizer of the objective function.
   */
  using y_t = typename optimizer_t::y_t;

  /**
   * @brief type of x.
   */
  using x_t = typename optimizer_t::x_t;

  /**
   * @brief Type for the hessian matrix of the objective function.
   */
  using y_hessian_t = typename optimizer_t::y_hessian_t;

  /**
   * @brief Type of the parameter vector
   */
  using params_t = typename optimizer_t::params_t;

  /**
   * @brief An interval of @c NUMERIC_TYPE
   */
  using interval_t = typename optimizer_t::interval_t;

  /**
   * @brief The type of the solution state
   */
  using solution_state_t = DAEOSolutionState<NUMERIC_T, XDIMS, YDIMS>;

  vector<solution_state_t> solve_daeo(NUMERIC_T t, NUMERIC_T t_end,
                                      NUMERIC_T dt, x_t const &x0,
                                      params_t const &params,
                                      std::string tag = "") {
    using clock = std::chrono::high_resolution_clock;
    vector<solution_state_t> solution_trajectory;
    DAEOSolverLogger logger(tag);
    if (m_settings.LOGGING_ENABLED) {
      logger.log_computation_begin(clock::now(), 0, t, dt, x0);
    }
    fmt::println("Starting to solve DAEO at t={:.4e}, x0={:.4e}, and dt={:.4e}",
                 t, x0, dt);
    typename optimizer_t::results_t opt_res =
        m_optimizer.find_minima_at(t, x0, params);
    fmt::println("  BNB optimizer yields candidates for y at {:::.4e}",
                 opt_res.minima_intervals);
    solution_state_t current, next;
    current = solution_state_from_optimizer_results(t, x0, opt_res, params);
    vector<y_t> dy;
    if (m_settings.LOGGING_ENABLED) {
      logger.log_global_optimization(clock::now(), 0, current.t, current.x,
                                     current.y, current.i_star);
    }

    // next portion relies on the assumption that two minimizers of h don't
    // "cross paths" inside of a time step even if they did, would it really
    // matter? since we don't do any implicit function silliness we probably
    // wouldn't even be able to tell if this did happen, but it may be
    // beneficial to periodically check all of the y_is and see if they're close
    // to each other before and after solving for the values at the next time
    // step. This would trigger a search for minima of h(x, y) again, since we
    // may have "lost" one

    size_t iter = 0, iterations_since_search = 0;
    while (current.t < t_end) {
      solution_trajectory.push_back(current);
      // we have not found an event in this time step (yet).
      bool event_found = false;
      next = integrate_daeo(current, dt, params);
      if (iterations_since_search == m_settings.SEARCH_FREQUENCY) {
        opt_res = m_optimizer.find_minima_at(next.t, next.x, params);
        if (opt_res.minima_intervals.size() == 0) {
          fmt::println(
              "*** SCREAMING CRYING VOMITING ON THE FLOOR iter={:d}***", iter);
          break;
        }
        solution_state_t from_opt = solution_state_from_optimizer_results(
            next.t, next.x, opt_res, params);
        if (m_settings.LOGGING_ENABLED) {
          logger.log_global_optimization(clock::now(), iter, from_opt.t,
                                         from_opt.x, from_opt.y,
                                         from_opt.i_star);
        }
        // check if we need to rewind multiple time steps
        fmt::println("  {:d} Checking identity of new optima at t={:.2e}", iter,
                     from_opt.t);
        fmt::println("  Current candidates for y are             {:::.4e}",
                     next.y);
        fmt::println("  BNB optimizer yields candidates for y at {:::.4e}",
                     from_opt.y);

        // do we use a flat threshold or a neighborhood criterion?
        if (from_opt.n_local_optima() <= next.n_local_optima()) {
          fmt::println(
              "  Optimizer may have vanished, from {:d} to {:d} optimizers.",
              next.n_local_optima(), from_opt.n_local_optima());
          fmt::println("  Reordering optima to match {:::.4e}", from_opt.y);
          if (m_settings.LINEARIZE_OPTIMIZER_DRIFT) {
            vector<NUMERIC_T> neighborhoods(from_opt.n_local_optima());
            std::transform(
                from_opt.y.begin(), from_opt.y.end(),
                neighborhoods.begin(), [&](const auto &y) -> auto{
                  return drift(from_opt.t, from_opt.x, y, params);
                });
            next = correct_optimizer_permutation(from_opt, next, neighborhoods);
          } else {
            next = correct_optimizer_permutation(
                from_opt, next, m_settings.EVENT_DRIFT_COEFF * dt);
          }
          fmt::println("               new order is {:::.4e}", next.y);
        } else {
          fmt::println("  Optimizer emerged.");
          fmt::println("  Reordered optima to match {:::.4e}", next.y);
          if (m_settings.LINEARIZE_OPTIMIZER_DRIFT) {
            vector<NUMERIC_T> neighborhoods(from_opt.n_local_optima());
            std::transform(
                from_opt.y.begin(), from_opt.y.end(),
                neighborhoods.begin(), [&](const auto &y) -> auto{
                  return drift(from_opt.t, from_opt.x, y, params);
                });
            from_opt =
                correct_optimizer_permutation(next, from_opt, neighborhoods);
          } else {
            from_opt = correct_optimizer_permutation(
                next, from_opt, m_settings.EVENT_DRIFT_COEFF * dt);
          }
          fmt::println("               new order is {:::.4e}", from_opt.y);
        }

        // check for event and correct any error that may have accumulated
        // from the local optimizer tracking
        event_found = (next.y_star() - from_opt.y_star()).norm() >
                      m_settings.EVENT_DETECTION_EPS;
        if (event_found)
          fmt::println("  event found by gopt");
        next = std::move(from_opt);
        iterations_since_search = 0;
      } else {
        // we avoid this simple check when gopt has run
        size_t min_idx = compute_global_minimizer(next, params);
        next.i_star = min_idx;
        event_found = next.i_star != current.i_star;
        if (event_found &&
            (m_objective.objective_value(next.t, next.x, next.y_star(),
                                         params) >=
             m_objective.objective_value(next.t, next.x, next.y[current.i_star],
                                         params))) {
          fmt::println("  wtf");
        }
      }

      // correct event if enabled in settings
      // we make the (bold) assumption that during a step containing an event,
      // no new local optimizers appear, and no current optimizers vanish
      // as long as the lost/gained optimizers are _not_ the global optimizer
      // at either end of the time step, this could be relaxed.

      if (m_settings.EVENT_DETECTION_AND_CORRECTION && event_found) {
        fmt::println(
            "  Detected global optimzer switch between (t={:.6e}, "
            "y={::.4e}, h={:.4e}) and (t={:.6e}, y={::.4e}, h={:.4e})",
            current.t, current.y_star(),
            m_objective.objective_value(current.t, current.x, current.y_star(),
                                        params),
            next.t, next.y_star(),
            m_objective.objective_value(next.t, next.x, next.y_star(), params));
        fmt::println("    Jump size is {:.6e}",
                     (current.y_star() - next.y_star()).norm());

        auto [computed_event, is_event] =
            locate_and_integrate_to_event_bisect(current, next, params);

        fmt::println("    Event at t={:.6e}, x={:.4e}", computed_event.t,
                     computed_event.x);

        // check if the event search actually yielded any change
        // by checking if the optimizer has shifted.
        if (!is_event) {
          // event was a misfire, something went wrong
          // unsure how this would happen, but it seems to quite often
          next.i_star = current.i_star;
          if (m_settings.LOGGING_ENABLED) {
            dy = compute_dy(current.y, next.y);
            logger.log_time_step(clock::now(), iter, next.t, dt, next.x,
                                 next.x - current.x, next.y, dy, next.i_star);
          }
        } else {
          NUMERIC_T dt_event = computed_event.t - current.t;
          NUMERIC_T dt_grid = next.t - computed_event.t;
          // project optimizers forward by integrating to event time
          solution_state_t at_event = integrate_daeo(current, dt_event, params);
          // update the relevant optimizers to the event
          // i.e. current global optimizer -> computed event's former global
          // optimizer and new global optimizer -> computed event's new global
          // optimizer these copies should be meaningless, since the integration
          // step is the same...
          // we should have this:
          // at_event.y[current.i_star] == computed_event.y[0];
          // at_event.y[next.i_star] == computed_event.y[1];
          // which means we can just do this
          at_event.i_star = next.i_star;
          if (m_settings.LOGGING_ENABLED) {
            dy = compute_dy(current.y, at_event.y);
            logger.log_event_correction(
                clock::now(), iter, at_event.t, dt_event, at_event.x,
                at_event.x - current.x, at_event.y, dy, at_event.i_star);
          }
          next = integrate_daeo(at_event, dt_grid, params);
          if (m_settings.LOGGING_ENABLED) {
            dy = compute_dy(at_event.y, next.y);
            logger.log_time_step(clock::now(), iter, next.t, dt, next.x,
                                 next.x - current.x, next.y, dy, next.i_star);
          }
        }
      } else {
        // we don't need to handle events, we can move on.
        if (m_settings.LOGGING_ENABLED) {
          dy = compute_dy(current.y, next.y);
          logger.log_time_step(clock::now(), iter, next.t, dt, next.x,
                               next.x - current.x, next.y, dy, next.i_star);
        }
      }

      current = std::move(next);
      iter++;
      iterations_since_search++;
    }
    if (m_settings.LOGGING_ENABLED) {
      logger.log_computation_end(clock::now(), iter, current.t, current.x,
                                 current.y, current.i_star);
    }
    return solution_trajectory;
  }

private:
  WrappedXPrimeFunction<XPRIME> m_xprime;
  WrappedObjective<OBJECTIVE> m_objective;
  DAEOSolverSettings<NUMERIC_T> const m_settings;
  optimizer_t m_optimizer;

  /**
   * @brief Create a valid solution state from the results of the global
   * optimizer at (t, x).
   */
  solution_state_t solution_state_from_optimizer_results(
      NUMERIC_T const t, x_t const &x,
      typename optimizer_t::results_t gopt_results, params_t p) {
    using boost::numeric::median;
    vector<y_t> y;
    for (auto &y_i : gopt_results.minima_intervals) {
      y.emplace_back(y_i.size());
      y.back() = y_i.unaryExpr([](auto ival) { return median(ival); });
    }
    solution_state_t ss{t, x, 0, y};
    if (ss.y.size() > 1) {
      ss.i_star = compute_global_minimizer(ss, p);
    }
    return ss;
  }

  /**
   * @brief Compute the index of the global optimizer @c y★ in a solution state
   * @c s given a parameter vector @c p .
   * @param[in] s The solution state state.
   * @param[in] p The parameter vector to pass through to the objective
   * function.
   * @returns @c i★ for the solution state.
   */
  size_t compute_global_minimizer(solution_state_t const &s,
                                  const params_t &p) {
    NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
    size_t i_star = 0;
    for (size_t i = 0; i < s.n_local_optima(); i++) {
      NUMERIC_T h = m_objective.objective_value(s.t, s.x, s.y[i], p);
      if (h < h_star) {
        h_star = h;
        i_star = i;
      }
    }
    return i_star;
  }

  vector<y_t> compute_dy(vector<y_t> const &y, vector<y_t> const &y_next) {
    vector<y_t> dy(y.size());
    for (size_t i = 0; i < y.size(); i++) {
      dy[i] = (y_next[i] - y[i]);
    }
    return dy;
  }

  /**
   * @brief Use the implicit function theorem to estimate dydt.
   */
  vector<y_t> estimate_dydt(solution_state_t const &s, params_t const &p) {
    auto dxdt = m_xprime.objective_value(s.t, s.x, s.y_star(), p);
    vector<y_t> res(s.y.size());
    for (size_t i = 0; i < s.y.size(); i++) {
      auto d2hdyx = m_objective.d2dxdy(s.t, s.x, s.y[i], p);
      y_hessian_t d2hdyy = m_objective.hess_y(s.t, s.x, s.y[i], p);
      auto dydx = d2hdyy.colPivHouseholderQr().solve(-1 * d2hdyx);
      res[i] = dydx * dxdt;
    }
    return res;
  }

  /**
   * @brief Take
   *
   *
   */
  solution_state_t correct_optimizer_permutation(solution_state_t const &few,
                                                 solution_state_t const &many,
                                                 NUMERIC_T const evt_thold) {
    vector<size_t> perm(few.n_local_optima());
    for (size_t i = 0; i < perm.size(); i++) {
      for (size_t j = 0; j < many.n_local_optima(); j++) {
        NUMERIC_T diff = (many.y[j] - few.y[i]).norm();
        if (diff < evt_thold) {
          perm[i] = j;
        }
      }
    }
    solution_state_t res = many;
    for (size_t i = 0; i < perm.size(); i++) {
      res.y[i] = many.y[perm[i]];
      if (perm[i] == many.i_star) {
        res.i_star = i;
      }
    }
    // if there are "leftovers" that weren't captured in the permutation vector
    if (few.n_local_optima() != many.n_local_optima()) {
      rearrange_leftovers(res, many, perm);
    }
    return res;
  }

  /**
   * @brief Take
   *
   *
   */
  solution_state_t
  correct_optimizer_permutation(solution_state_t const &few,
                                solution_state_t const &many,
                                vector<NUMERIC_T> const neighborhoods) {
    vector<size_t> perm(few.n_local_optima());
    for (size_t i = 0; i < perm.size(); i++) {
      for (size_t j = 0; j < many.n_local_optima(); j++) {
        NUMERIC_T diff = (many.y[j] - few.y[i]).norm();
        if (diff < neighborhoods[i]) {
          perm[i] = j;
        }
      }
    }
    solution_state_t res = many;
    for (size_t i = 0; i < perm.size(); i++) {
      res.y[i] = many.y[perm[i]];
      if (perm[i] == many.i_star) {
        res.i_star = i;
      }
    }
    // if there are "leftovers" that weren't captured in the permutation vector
    if (few.n_local_optima() != many.n_local_optima()) {
      rearrange_leftovers(res, many, perm);
    }
    return res;
  }

  /**
   * @brief
   *
   */
  void rearrange_leftovers(solution_state_t &res, solution_state_t const &many,
                           vector<size_t> const &perm) {
    vector<size_t> leftovers;
    for (size_t i = 0; i < many.n_local_optima(); i++) {
      bool leftover = true;
      for (size_t j = 0; (j < perm.size() && leftover); j++) {
        leftover = leftover && (perm[j] != i);
      }
      if (leftover) {
        leftovers.push_back(i);
      }
    }
    for (size_t i = 0; i < leftovers.size(); i++) {
      res.y[perm.size() + i] = many.y[leftovers[i]];
      if (leftovers[i] == many.i_star) {
        res.i_star = perm.size() + i;
      }
    }
  }

  /*
      G and delG are used by newton iteration to find x and y at the next time
     step. There is the condition on x from the trapezoidal rule: x_{k+1} =
     x_{k} + dt/2 * (f(x_{k+1}, y_{k+1}) + f(x_{k}, y_{k})) After the zeroth
     time step (t=t0), we have good initial guesses for the minima y_{k+1}
      (assuming they don't drift "too far")
      The other equations are provided by dh/dyi = 0 at x_{k+1} and y_{k+1}

      from this structure we compute G and delG in blocks, since we have
     derivative routines available for f=x' and h
  */

  /**
   * @brief Compute G. We are searching for x1, y1 s.t. G(...) = 0.
   * @param[in] start Value of the solution trajectory at the beginning of a
   * time step.
   * @param[in] end Value (potentially a guess) of the solution trajectory at
   * the end of a time step.
   * @param[in] dt Current time step size.
   * @param[in] p Parameter vector.
   */
  Eigen::VectorX<NUMERIC_T> trapezoidal_rule(solution_state_t const &start,
                                             solution_state_t const &guess,
                                             NUMERIC_T const dt,
                                             params_t const &p) {
    int ydims = start.ydims();
    // fix i_star (assume no event in this part of the program)
    size_t i_star = start.i_star;
    int gdims = start.xdims() + start.n_local_optima() * ydims;
    Eigen::VectorX<NUMERIC_T> result(gdims);
    result(Eigen::seq(0, start.xdims())) =
        start.x - guess.x +
        dt / 2 *
            (m_xprime.xprime_value(start.t, start.x, start.y[i_star], p) +
             m_xprime.xprime_value(guess.t, guess.x, guess.y[i_star], p));
    for (int i = 0; i < start.n_local_optima(); i++) {
      result(Eigen::seqN(1 + i * ydims, ydims)) =
          m_objective.grad_y(guess.t, guess.x, guess.y[i], p);
    }
    return result;
  }

  /**
   * @brief Gradient of the function used for newton iteration.
   * @param[in] guess The guessed value of the solution trajectory.
   * @param[in] dt The time step size from the beginning of the time step to
   * where @c guess is evaluated.
   * @param[in] p Parameter vector.
   */
  Eigen::MatrixX<NUMERIC_T>
  trapezoidal_rule_jacobian(solution_state_t const &guess, NUMERIC_T const dt,
                            params_t const &p) {
    using Eigen::seqN;
    int ydims = guess.ydims();
    int xdims = guess.xdims();
    int ndims = xdims + guess.n_local_optima() * ydims;
    Eigen::MatrixX<NUMERIC_T> result(ndims, ndims);
    result(0, 0) =
        -1 +
        dt / 2 * m_xprime.grad_x(guess.t, guess.x, guess.y[guess.i_star], p);
    for (size_t i = 0; i < guess.n_local_optima(); i++) {
      result(0, seqN(1 + i * ydims, ydims)) =
          dt / 2 * m_xprime.grad_y(guess.t, guess.x, guess.y[i], p);
      result(seqN(1 + i * ydims, ydims), 0) =
          m_objective.d2dxdy(guess.t, guess.x, guess.y[i], p);
      result(seqN(1 + i * ydims, ydims), seqN(1 + i * ydims, ydims)) =
          m_objective.hess_y(guess.t, guess.x, guess.y[i], p);
    }
    return result;
  }

  /**
   * @brief Integrate the  DAEO from time @c t to @c t+dt
   * @param[in] start The value of the solution at @c t=t0
   * @param[in] dt The size of the time step to integrate.
   * @param[in] p Parameter vector.
   * @return The value of solution at @c t=t+dt.
   * @details Integrates the ODE using the trapezoidal rule. Additionally solves
   * ∂h/∂y_k = 0 simultaenously using Newton's method.
   * DOES @b NOT UPDATE THE THE RESULT'S I_STAR!
   */
  solution_state_t integrate_daeo(solution_state_t const &start, NUMERIC_T dt,
                                  params_t const &p) {
    // copy and make a guess using an estimate of dydt
    solution_state_t next(start);
    vector<y_t> dydt = estimate_dydt(start, p);
    next.t += dt;
    for (size_t i = 0; i < next.y.size(); i++) {
      next.y[i] += dydt[i] * dt;
    }

    Eigen::VectorX<NUMERIC_T> G, diff;
    Eigen::MatrixX<NUMERIC_T> jacG;
    size_t iter = 0;
    while (iter < m_settings.MAX_NEWTON_ITERATIONS) {
      G = trapezoidal_rule(start, next, dt, p);
      jacG = trapezoidal_rule_jacobian(next, dt, p);
      diff = jacG.colPivHouseholderQr().solve(G);
      next.x = next.x - diff(0);
      for (size_t i = 0; i < start.n_local_optima(); i++) {
        next.y[i] =
            next.y[i] - diff(Eigen::seqN(1 + i * start.ydims(), start.ydims()));
      }
      if (diff.norm() < m_settings.NEWTON_EPS) {
        break;
      }
      iter++;
    }
    return next;
  }

  /*
      H and dHdt are used to find the precise event location
      either by using newton iteration or the bisection method.
  */

  /**
   * @brief Event function between optima @c y1 (current optimum at (t, x))
   *   and the candidate next optimum @c y2 at (t, x)
   * @param[in] t
   * @param[in] x
   * @param[in] y1 Global optimizer from the beginning of the time step,
   * integrated to @c t
   * @param[in] y2 Global optimizer at the end of the time step, integrated to
   * @c t
   * @param[in] p Parameter vector.
   */
  inline NUMERIC_T event_function(NUMERIC_T t, NUMERIC_T x, y_t const &y1,
                                  y_t const &y2, params_t const &p) {
    return m_objective.objective_value(t, x, y1, p) -
           m_objective.objective_value(t, x, y2, p);
  }

  /**
   * @brief @b TOTAL derivative of @c H w.r.t. @c x
   */
  inline NUMERIC_T event_function_ddx(NUMERIC_T t, NUMERIC_T x, y_t const &y1,
                                      y_t const &y2, params_t const &p) {
    return m_objective.grad_x(t, x, y1, p) - m_objective.grad_x(t, x, y2, p);
  }

  /**
   * @deprecated
   * @brief Locate and correct an event between @c start and @c end.
   * Uses Newton's method, which has a tendency to escape the interval in
   * question.
   *
   * @param[in] start The solution state at the beginning of the time step.
   * @param[in] end The (incorrect) solution state at the end of the time step.
   * @param[in] p Parameter vector.
   * @return Value of solution at event.
   *
   * @details We assume that no optimizers emerge or vanish inside this time
   * step. This allows us to assume that @c start.y_star() and @c end.y_star()
   * both exist as possible optimizers for the duration of the time step, even
   * if we are only provided with the results from global optimization.
   */
  solution_state_t
  locate_and_integrate_to_event_newton(solution_state_t const &start,
                                       solution_state_t const &end,
                                       params_t const &p) {

    // make local copies that only know about the swapped optimizer
    solution_state_t left{start.t, start.x, 0, {start.y_star(), end.y_star()}};
    solution_state_t right{end.t, end.x, 1, {start.y_star(), end.y_star()}};
    NUMERIC_T H, dHdt, dt_guess;
    solution_state_t guess;
    dt_guess = (right.t - left.t) / 2;
    size_t iter = 0;
    bool escaped = false;
    while (iter < m_settings.MAX_NEWTON_ITERATIONS) {
      // integrate to t_guess
      guess = integrate_daeo(left, dt_guess, p);
      // evaluate event function
      H = event_function(guess.t, guess.x, guess.y[left.i_star],
                         guess.y[right.i_star], p);

      if (fabs(H) < m_settings.NEWTON_EPS) {
        break;
      }
      // only scream once
      if (!escaped && (left.t > guess.t || guess.t > right.t)) {
        fmt::println("  Escaped bounds on event locator!!");
        escaped = true;
        // breaking may save us here.
        break;
      }

      guess.i_star = compute_global_minimizer(guess, p);
      dHdt = (event_function_ddx(guess.t, guess.x, guess.y[left.i_star],
                                 guess.y[right.i_star], p) *
              m_xprime.objective_value(guess.t, guess.x, guess.y_star(), p));
      // dHdt += partial h partial t at y1 and y2... maybe not necessary.
      // newton iteration.
      guess.t -= H / dHdt;
      dt_guess = guess.t - start.t;
      iter++;
    }
    return guess;
  }

  /**
   * @brief Locate and correct an event that happens between @c start and @c end
   * using bisection.
   * @param[in] start The solution state at the beginning of the time step.
   * @param[in] end The (incorrect) solution state at the end of the time step.
   * @param[in] p Parameter vector.
   * @return Value of solution at event. Result @c i_star will be equal to one
   * if an event was located, zero otherwise.
   *
   * @details We assume that no optimizers emerge or vanish inside this time
   * step. This allows us to assume that @c start.y_star() and @c end.y_star()
   * both exist as possible optimizers for the duration of the time step, even
   * if we are only provided with the results from global optimization.
   */
  std::tuple<solution_state_t, bool>
  locate_and_integrate_to_event_bisect(solution_state_t const &start,
                                       solution_state_t const &end,
                                       params_t const &p) {
    // make local copies that only know about the swapped optimizer
    solution_state_t left{
        start.t, start.x, 0, {start.y_star(), start.y[end.i_star]}};
    solution_state_t right{
        end.t, end.x, 1, {end.y[start.i_star], end.y_star()}};
    solution_state_t guess;
    NUMERIC_T delta = (end.t - start.t) / 2;
    NUMERIC_T dt_guess = delta;
    NUMERIC_T H;
    size_t iter = 0;
    // Newton gains "one digit" of accuracy for every iteration (in nice cases)
    // Bisection gains one power of 2 every iteration
    // we should never need more iterations than for the worst-case newton
    // solver here
    while (iter < m_settings.MAX_NEWTON_ITERATIONS) {
      delta = delta / 2; // Bisect.
      guess = integrate_daeo(left, dt_guess, p);
      H = event_function(guess.t, guess.x, guess.y[left.i_star],
                         guess.y[right.i_star], p);
      // maybe we need to choose a different tolerance for bisection, but using
      // the same number of digits of accuracy for bisection and newton seems to
      // work.
      if (fabs(H) < m_settings.NEWTON_EPS) {
        break;
      } else if (H < 0) {
        dt_guess += delta;
      } else {
        dt_guess -= delta;
      }
      iter++;
    }
    guess.i_star = 1;
    return {guess, (fabs(H) < m_settings.NEWTON_EPS)};
  }
};

#endif