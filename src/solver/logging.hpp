/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Logging utilities for the solver and optimizer.
 */
#ifndef _SOLVER_LOGGING_HPP
#define _SOLVER_LOGGING_HPP

#ifndef DATA_OUTPUT_DIR
#define DATA_OUTPUT_DIR .
#endif // data output directory defined?

#include <chrono>
#include <fstream>
#include <ranges>
#include <string>
#include <vector>

#include "fmt/chrono.h"
#include "fmt/core.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

#include "utils/daeo_utils.hpp"

using std::vector;

enum OptimizerEventCode {
  OPTIMIZATION_BEGIN,
  OPTIMIZATION_COMPLETE,
  TASK_BEGIN,
  TASK_COMPLETE,
  VALUE_TEST,
  CONVERGENCE_TEST,
  GRADIENT_TEST,
  HESSIAN_TEST,
  ALL_TESTS
};

inline auto format_as(OptimizerEventCode evc) { return fmt::underlying(evc); }

constexpr char OPTIMIZER_LOG_COLUMNS[]{
    "TASKNUM\tTSTAMP\tEVENTID\tEXTRACODE\tX\tH\tDHDX\tD2HDX2\tCONVERGENCE"};
constexpr char LOG_TNUM_TSTAMP[]{"{:d}\t{}\t"};
constexpr char LOG_EID_EXTRA[]{"{:d}\t{:d}\t"};
constexpr char LOG_NUMERIC_VAL[]{"{:.8e}\t"};
constexpr char LOG_ITERABLE_NUMERIC_VALS[]{"{::.8e}\t"};
constexpr char LOG_MATRIX_NUMERIC_VALS[]{"{:::.8e}\t"};

using sys_time_point_t = std::chrono::time_point<std::chrono::system_clock>;
class BNBOptimizerLogger {
  size_t m_threadcount;
  sys_time_point_t m_logging_start;
  vector<std::ofstream> outs;

public:
  BNBOptimizerLogger(std::string const &tag) : m_threadcount{1} {
    outs.emplace_back(
        fmt::format("{}/{}_optimizer_thread_0.tsv", DATA_OUTPUT_DIR, tag));
    fmt::println(outs[0], OPTIMIZER_LOG_COLUMNS);
  };

  BNBOptimizerLogger(size_t t_threads, std::string const &tag)
      : m_threadcount{t_threads} {
    for (size_t i = 0; i < m_threadcount; i++) {
      outs.emplace_back(fmt::format("{}/{}_optimizer_thread_{}.tsv",
                                    DATA_OUTPUT_DIR, tag, i));
      fmt::println(outs[i], OPTIMIZER_LOG_COLUMNS);
    }
  };

  ~BNBOptimizerLogger() {
    for (size_t i = 0; i < m_threadcount; i++) {
      outs[i].close();
    }
  }

  template <std::ranges::range Y>
  void log_computation_begin(sys_time_point_t time, size_t tasknum,
                             Y const &domain, size_t threadid = 0) {
    m_logging_start = time;
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, 0, 0);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, domain);
    fmt::print(outs[threadid], "\t\t\t\n");
  }

  void log_computation_end(sys_time_point_t time, size_t tasknum,
                           size_t n_results, size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, OPTIMIZATION_COMPLETE, n_results);
    fmt::print(outs[threadid], "\t");
    fmt::print(outs[threadid], "\t\t\t\n");
  }

  template <std::ranges::range Y>
  void log_task_begin(sys_time_point_t time, size_t tasknum, Y const &y,
                      size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, TASK_BEGIN, 0);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
    fmt::print(outs[threadid], "\t\t\t\n");
  }

  template <std::ranges::range Y>
  void log_task_complete(sys_time_point_t time, size_t tasknum, Y const &y,
                         size_t reason, size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, TASK_COMPLETE, reason);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
    fmt::print(outs[threadid], "\t\t\t\n");
  }

  void log_convergence_test(sys_time_point_t time, size_t tasknum,
                            vector<bool> const &convergence,
                            size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, CONVERGENCE_TEST, 0);
    fmt::print(outs[threadid], "\t\t\t\t{::d}\n", convergence);
  }

  template <typename T, std::ranges::range Y, std::ranges::range DHDY>
  void log_gradient_test(sys_time_point_t time, size_t tasknum, Y const &y,
                         T const &h, DHDY const &dhdy, size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, GRADIENT_TEST, 0);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
    fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, dhdy);
    fmt::print(outs[threadid], "\t\n");
  }

  /**
   * Log the result of the Hessian test.
   */
  template <typename T, std::ranges::range Y, std::ranges::range DHDY,
            std::ranges::range DDHDDY_ROWS>
  void log_hessian_test(sys_time_point_t time, size_t tasknum, size_t testres,
                        Y const &y, T const &h, DHDY const &dhdy,
                        DDHDDY_ROWS &d2hdy2, size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, HESSIAN_TEST, testres);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
    fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, dhdy);
    fmt::print(outs[threadid], LOG_MATRIX_NUMERIC_VALS, d2hdy2);
    fmt::print(outs[threadid], "\n");
  }

  /**
   * @brief Log results from all tests. Pass the Hessian as an iterator of rows
   * via (...).rowwise().
   */
  template <typename T, std::ranges::range Y, std::ranges::range DHDY,
            std::ranges::range DDHDYY_ROWS>
  void log_all_tests(sys_time_point_t time, size_t tasknum,
                     size_t combined_results, Y const &y, T const &h,
                     DHDY const &dhdy, DDHDYY_ROWS const &d2hdy2,
                     vector<bool> const &convergence, size_t threadid = 0) {
    std::chrono::duration<double, std::micro> deltaT = time - m_logging_start;
    fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, deltaT);
    fmt::print(outs[threadid], LOG_EID_EXTRA, ALL_TESTS, combined_results);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
    fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
    fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, dhdy);
    fmt::print(outs[threadid], LOG_MATRIX_NUMERIC_VALS, d2hdy2);
    fmt::print(outs[threadid], "{::d}\n", convergence);
  }
};

enum SolverEventCode {
  SOLVER_BEGIN,
  SOLVER_COMPLETE,
  TIME_STEP_NO_EVENT,
  TIME_STEP_EVENT_CORRECTED,
  OPTIMIZE,
  OPTIMUM_CHANGE
};

inline auto format_as(SolverEventCode evc) { return fmt::underlying(evc); }

constexpr char SOLVER_LOG_COLUMNS[]{
    "ITERATION\tTSTAMP\tEVENTID\tEXTRACODE\tT\tDT\tX\tDX\tY\tDY\tISTAR"};
constexpr char LOG_INTEGER_VAL[]{"{:d}"};

class DAEOSolverLogger {
  sys_time_point_t m_logging_start;
  std::ofstream out;

public:
  DAEOSolverLogger(std::string const &tag)
      : out(fmt::format("{}/{}_solver_log.tsv", DATA_OUTPUT_DIR, tag)) {
    fmt::println(out, SOLVER_LOG_COLUMNS);
  }

  ~DAEOSolverLogger() { out.close(); }

  template <typename T>
  void log_computation_begin(sys_time_point_t const tstamp, size_t const iter,
                             T const t0, T const dt0, T const x0) {
    m_logging_start = tstamp;
    std::chrono::duration<double, std::micro> deltaT = tstamp - m_logging_start;
    fmt::print(out, LOG_TNUM_TSTAMP, iter, deltaT);
    fmt::print(out, LOG_EID_EXTRA, SOLVER_BEGIN, 0);
    fmt::print(out, LOG_NUMERIC_VAL, t0);
    fmt::print(out, LOG_NUMERIC_VAL, dt0);
    fmt::print(out, LOG_NUMERIC_VAL, x0);
    // no dx, no y, no dy, no i_star
    fmt::print(out, "\t\t\t0\n");
  }

  template <typename T, std::ranges::range Y>
  void log_computation_end(sys_time_point_t const tstamp, size_t const iter,
                           T const t, T const x, Y const &y,
                           size_t const i_star) {
    std::chrono::duration<double, std::micro> deltaT = tstamp - m_logging_start;
    fmt::print(out, LOG_TNUM_TSTAMP, iter, deltaT);
    fmt::print(out, LOG_EID_EXTRA, SOLVER_COMPLETE, 0);
    fmt::print(out, LOG_NUMERIC_VAL, t);
    // no dt
    fmt::print(out, "\t");
    fmt::print(out, LOG_NUMERIC_VAL, x);
    // no dx
    fmt::print(out, "\t");
    fmt::print(out, LOG_MATRIX_NUMERIC_VALS, y);
    // no dy
    fmt::print(out, "\t");
    fmt::println(out, LOG_INTEGER_VAL, i_star);
  }

  template <typename T, std::ranges::range Y>
  void log_global_optimization(sys_time_point_t const tstamp, size_t const iter,
                               T const t, T const x, Y const &y,
                               size_t const i_star) {
    std::chrono::duration<double, std::micro> deltaT = tstamp - m_logging_start;
    fmt::print(out, LOG_TNUM_TSTAMP, iter, deltaT);
    fmt::print(out, LOG_EID_EXTRA, OPTIMIZE, 0);
    fmt::print(out, LOG_NUMERIC_VAL, t);
    // no dt
    fmt::print(out, "\t");
    fmt::print(out, LOG_NUMERIC_VAL, x);
    // no dx
    fmt::print(out, "\t");
    fmt::print(out, LOG_MATRIX_NUMERIC_VALS, y);
    // no dy
    fmt::print(out, "\t");
    fmt::println(out, LOG_INTEGER_VAL, i_star);
  }

  template <typename T, std::ranges::range Y>
  void log_time_step(sys_time_point_t const tstamp, size_t const iter,
                     T const t, T const dt, T const x, T const dx, Y const &y,
                     Y const &dydt, size_t const i_star) {
    std::chrono::duration<double, std::micro> deltaT = tstamp - m_logging_start;
    fmt::print(out, LOG_TNUM_TSTAMP, iter, deltaT);
    fmt::print(out, LOG_EID_EXTRA, TIME_STEP_NO_EVENT, 0);
    fmt::print(out, LOG_NUMERIC_VAL, t);
    fmt::print(out, LOG_NUMERIC_VAL, dt);
    fmt::print(out, LOG_NUMERIC_VAL, x);
    fmt::print(out, LOG_NUMERIC_VAL, dx);
    fmt::print(out, LOG_MATRIX_NUMERIC_VALS, y);
    fmt::print(out, LOG_MATRIX_NUMERIC_VALS, dydt);
    fmt::println(out, LOG_INTEGER_VAL, i_star);
  }

  template <typename T, std::ranges::range Y>
  void log_event_correction(sys_time_point_t const tstamp, size_t const iter,
                            T const t, T const dt, T const x, T const dx,
                            Y const &y, Y const &dydt, size_t const i_star) {
    std::chrono::duration<double, std::micro> deltaT = tstamp - m_logging_start;
    fmt::print(out, LOG_TNUM_TSTAMP, iter, deltaT);
    fmt::print(out, LOG_EID_EXTRA, TIME_STEP_EVENT_CORRECTED, 0);
    fmt::print(out, LOG_NUMERIC_VAL, t);
    fmt::print(out, LOG_NUMERIC_VAL, dt);
    fmt::print(out, LOG_NUMERIC_VAL, x);
    fmt::print(out, LOG_NUMERIC_VAL, dx);
    fmt::print(out, LOG_MATRIX_NUMERIC_VALS, y);
    fmt::print(out, LOG_MATRIX_NUMERIC_VALS, dydt);
    fmt::println(out, LOG_INTEGER_VAL, i_star);
  }
};
#endif