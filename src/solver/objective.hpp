/**
 * @file objective.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Wrapper classes for functions of the form f(t, x, y, p).
 */
#ifndef _OBJ_FUNCTION_WRAPPER_HPP
#define _OBJ_FUNCTION_WRAPPER_HPP

#include <type_traits>

// It's recommended to include eigen/boost before DCO
#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"
#include "dco.hpp"

#include "utils/daeo_utils.hpp"

template <typename T> struct is_boost_interval : std::false_type {};

template <typename T, typename POLICIES>
struct is_boost_interval<boost::numeric::interval<T, POLICIES>>
    : std::true_type {};

template <typename T, int ROWS, int COLS>
struct is_boost_interval<Eigen::Matrix<T, ROWS, COLS>> : is_boost_interval<T> {};

template <typename T>
concept IsInterval = is_boost_interval<T>::value;

template <typename FN, typename T, typename X, typename Y, int XDIMS, int YDIMS,
          int PDIMS>
concept PreservesIntervalsX =
    (!IsInterval<X>) || requires(FN f, T t, Eigen::Vector<X, XDIMS> const &x,
                                 Eigen::Vector<Y, YDIMS> const &y,
                                 Eigen::Vector<T, PDIMS> const &p) {
      { f(t, x, y, p) } -> IsInterval;
    };
template <typename FN, typename T, typename X, typename Y, int XDIMS, int YDIMS,
          int PDIMS>
concept PreservesIntervalsY =
    (!IsInterval<Y>) || requires(FN f, T t, Eigen::Vector<X, XDIMS> const &x,
                                 Eigen::Vector<Y, YDIMS> const &y,
                                 Eigen::Vector<T, PDIMS> const &p) {
      { f(t, x, y, p) } -> IsInterval;
    };

template <typename FN, typename T, typename X, typename Y, int XDIMS, int YDIMS,
          int PDIMS>
concept PreservesIntervals =
    PreservesIntervalsX<FN, T, X, Y, XDIMS, YDIMS, PDIMS> &&
    PreservesIntervalsY<FN, T, X, Y, XDIMS, YDIMS, PDIMS>;

/**
 * @brief Wraps a function of the form f(t, x, y, p) for use with the
 * optimizer and solver. Assumes that the return type of f is a scalar and
 * matches the type of scalar y.
 */
template <typename FN> class WrappedObjective {

  mutable size_t n_h_evaluations = 0;
  mutable size_t n_dy_evaluations = 0;
  mutable size_t n_d2y_evaluations = 0;
  mutable size_t n_dx_evaluations = 0;
  mutable size_t n_d2xy_evaluations = 0;

  // This return type is clumsy. Need to figure out a way to express
  // "promote this to a dco type, otherwise promote to an interval, otherwise
  // return a passive numerical value"

  /**
   * @brief The function to wrap and augment with partial derivative routines.
   */
  FN const fn;

public:
  WrappedObjective(FN const &t_fn) : fn{t_fn} {};

  /**
   * @brief Evaluate @c m_fn at the provided arguments.
   * @returns Value of @c m_fn .
   */
  template <typename NUMERIC_T, typename XT, typename YT, int XDIMS, int YDIMS,
            int PDIMS>
    requires PreservesIntervals<FN, NUMERIC_T, XT, YT, XDIMS, YDIMS, PDIMS>
  auto objective_value(NUMERIC_T const t, Eigen::Vector<XT, XDIMS> const &x,
                       Eigen::Vector<YT, YDIMS> const &y,
                       Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
      -> decltype(fn(t, x, y, p)) {
    n_h_evaluations += 1;
    return fn(t, x, y, p);
  }

  template <typename T, typename X, typename Y_ACTIVE_T, int XDIMS, int YDIMS,
            int PDIMS>
  auto grad_y(T const t, Eigen::Vector<X, XDIMS> const &x,
              Eigen::Vector<Y_ACTIVE_T, YDIMS> const &y,
              Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Vector<decltype(fn(t, x, y, p)), YDIMS> {
    n_dy_evaluations += 1;
    using dco_mode_t = dco::gt1s<Y_ACTIVE_T>;
    using active_t = typename dco_mode_t::type;
    // active inputs
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    // no vector assignment routines are available for eigen+dco
    // (until dco base 4.2)
    for (int i = 0; i < y.rows(); i++) {
      dco::value(y_active(i)) = y(i);
    }
    // and active outputs
    active_t h_active;
    // harvest derivative
    Eigen::Vector<Y_ACTIVE_T, YDIMS> dhdy(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::derivative(y_active(i)) = 1;
      h_active = fn(t, x, y_active, p);
      dhdy(i) = dco::derivative(h_active);
      dco::derivative(y_active(i)) = 0;
    }
    return dhdy;
  }

  template <typename T, typename X_ACTIVE_T, typename Y, int XDIMS, int YDIMS,
            int PDIMS>
  auto grad_x(T const t, Eigen::Vector<X_ACTIVE_T, XDIMS> const &x,
              Eigen::Vector<Y, YDIMS> const &y,
              Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Vector<decltype(fn(t, x, y, p)), XDIMS> {
    n_dx_evaluations += 1;
    using dco_mode_t = dco::gt1s<X_ACTIVE_T>;
    using active_t = typename dco_mode_t::type;
    Eigen::Vector<active_t, XDIMS> x_active(x.rows());
    for (int i = 0; i < x.rows(); i++) {
      dco::value(x_active(i)) = x(i);
    }
    Eigen::Vector<X_ACTIVE_T, XDIMS> dhdx(x.rows());
    active_t h_active;
    for (int i = 0; i < x.rows(); i++) {
      dco::derivative(x_active(i)) = 1;
      h_active = fn(t, x_active, y, p);
      dhdx(i) = dco::derivative(h_active);
      dco::derivative(x_active(i)) = 0;
    }
    return dhdx;
  }

  template <typename T, typename X, typename Y_ACTIVE_T, int XDIMS, int YDIMS,
            int PDIMS>
  auto hess_y(T const t, Eigen::Vector<X, XDIMS> const &x,
              Eigen::Vector<Y_ACTIVE_T, YDIMS> const &y,
              Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Matrix<decltype(fn(t, x, y, p)), YDIMS, YDIMS> {
    n_d2y_evaluations += 1;
    using dco_tangent_t = typename dco::gt1s<Y_ACTIVE_T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();

    // active inputs
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    auto start_position = tape->get_position();

    // active outputs
    active_t h_active;
    // Hessian of a scalar function is a symmetric square matrix
    // (provided second derivative symmetry holds)
    Eigen::Matrix<Y_ACTIVE_T, YDIMS, YDIMS> d2hdy2(y.rows(), y.rows());
    for (int hrow = 0; hrow < y.rows(); hrow++) {
      dco::derivative(dco::value(y_active(hrow))) = 1; // wiggle y[hrow]
      h_active = fn(t, x, y_active, p);        // compute h
      dco::value(dco::derivative(h_active)) = 1;
      tape->interpret_adjoint_and_reset_to(start_position);
      for (int hcol = 0; hcol < y.rows(); hcol++) {
        d2hdy2(hrow, hcol) = dco::derivative(dco::derivative(y_active(hcol)));
        // reset any accumulated values
        dco::derivative(dco::derivative(y_active(hcol))) = 0;
        dco::value(dco::derivative(y_active(hcol))) = 0;
      }
      // no longer wiggling y[hrow]
      dco::derivative(dco::value(y_active(hrow))) = 0;
    }
    return d2hdy2;
  }

  template <typename T, typename XY_ACTIVE_T, int XDIMS, int YDIMS, int PDIMS>
  auto d2dxdy(T const t, Eigen::Vector<XY_ACTIVE_T, XDIMS> const &x,
              Eigen::Vector<XY_ACTIVE_T, YDIMS> const &y,
              Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Matrix<decltype(fn(t, x, y, p)), XDIMS, YDIMS> {
    n_d2xy_evaluations += 1;
    using dco_tangent_t = typename dco::gt1s<XY_ACTIVE_T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    active_t h_active;
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    Eigen::Vector<active_t, XDIMS> x_active(x.rows());
    for (int i = 0; i < x.rows(); x++) {
      dco::passive_value(x_active(i)) = x(i);
      tape->register_variable(x_active(i));
    }
    Eigen::Matrix<XY_ACTIVE_T, XDIMS, YDIMS> ddxddy(x.rows(), y.rows());
    for (int i = 0; i < x_active.rows(); i++) {
      dco::derivative(dco::value(x_active(i))) = 1;    // wiggle x(i)
      h_active = fn(t, x_active, y_active, p); // compute h
      dco::value(dco::derivative(h_active)) = 1;       // sensitivity to h is 1
      tape->interpret_adjoint();
      // harvest derivative
      for (int j = 0; j < ddxddy.rows(); j++) {
        ddxddy(i, j) =
            dco::derivative(dco::derivative(y_active(j))); // harvest d2dxdy
        // reset any accumulated values
        dco::derivative(dco::derivative(y_active(j))) = 0;
        dco::value(dco::derivative(y_active(j))) = 0;
      }
      dco::derivative(dco::derivative(x_active(i))) = 0;
      dco::derivative(dco::value(x_active(i))) = 0;
    }

    return ddxddy;
  }

  /**
   * @brief Returns, as a tuple of 5 elements, the number of calls to each of
   * the provided drivers.
   */
  ntuple<5, size_t> statistics() const {
    return {n_h_evaluations, n_dy_evaluations, n_dx_evaluations,
            n_d2y_evaluations, n_d2xy_evaluations};
  }
};

// IDEA
// we only really care about the behavior behind operator()
// could we do something like this and then use L(...) to verify the hessian
// test? could be worth a look!
template <typename F, typename G> class DAEOWrappedConstrained {

  /**
   * @brief The wrapped objective function.
   */
  F m_objective;

  /**
   * @brief The wrapped constraint.
   */
  G m_constraint;

  mutable size_t n_h_evaluations = 0;
  mutable size_t n_dhdy_evaluations = 0;
  mutable size_t n_d2hdy2_evaluations = 0;
  mutable size_t n_dhdx_evaluations = 0;
  mutable size_t n_d2hdxdy_evaluations = 0;

  mutable size_t n_L_evaluations = 0;
  mutable size_t n_dLdy_evaluations = 0;

  // This return type is clumsy. Need to figure out a way to express
  // "promote this to a dco type, otherwise promote to an interval, otherwise
  // return a passive numerical value"
public:
  template <typename T, typename XT, typename YT, int YDIMS, int PDIMS>
    requires PreservesIntervals<F, T, XT, YT, 1, YDIMS, PDIMS>
  auto objective_value(T const t, XT const x, Eigen::Vector<YT, YDIMS> const &y,
                       Eigen::Vector<T, PDIMS> const &p) const
      -> decltype(m_objective(t, x, y, p)) {
    n_h_evaluations += 1;
    return m_objective(t, x, y, p);
  }

  template <typename T, typename XT, typename YT, int YDIMS_EXT, int PDIMS>
    requires PreservesIntervals<F, T, XT, YT, 1, YDIMS_EXT, PDIMS> &&
                 PreservesIntervals<G, T, XT, YT, 1, YDIMS_EXT, PDIMS>
  auto lagrangian_value(T const t, XT const x,
                        Eigen::Vector<YT, YDIMS_EXT> const &y_ext,
                        Eigen::Vector<T, PDIMS> const &p) const
      -> decltype(m_objective(t, x, y_ext, p)) {
    using Eigen::seq, Eigen::placeholders::last;
    n_L_evaluations += 1;
    return m_objective(t, x, y_ext(seq(1, last)), p) +
           y_ext(0) * m_constraint(t, x, y_ext(seq(1, last)), p);
  }

  template <typename T, typename X, typename Y_ACTIVE_T, int YDIMS_EXT,
            int PDIMS>
  auto grad_y_lagrangian(T const t, X const x,
                         Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> const &y,
                         Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Vector<decltype(m_objective(t, x, y, p)), YDIMS_EXT> {
    n_dhdy_evaluations += 1;
    // define dco types and get a pointer to the tape
    // unsure how to use ga1sm to expand this to multithreaded programs
    using dco_mode_t = dco::ga1s<Y_ACTIVE_T>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();
    Eigen::Vector<active_t, YDIMS_EXT> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    active_t L_active = lagrangian_value(t, x, y_active, p);
    tape->register_output_variable(L_active);
    dco::derivative(L_active) = 1;
    tape->interpret_adjoint();
    // harvest derivative
    Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> dLdy(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dLdy(i) = dco::derivative(y_active(i));
    }
    return dLdy;
  }

  template <typename T, typename X, typename Y_ACTIVE_T, int YDIMS_EXT,
            int PDIMS>
  auto hess_y_lagrangian(T const t, X const x,
                         Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> const &y,
                         Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Matrix<decltype(m_objective(t, x, y, p)), YDIMS_EXT,
                       YDIMS_EXT> {
    n_d2hdy2_evaluations += 1;

    using dco_tangent_t = typename dco::gt1s<Y_ACTIVE_T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();

    active_t L_active;
    Eigen::Vector<active_t, YDIMS_EXT> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    auto start_position = tape->get_position();
    Eigen::Matrix<Y_ACTIVE_T, YDIMS_EXT, YDIMS_EXT> d2Ldy2(y.rows(), y.rows());
    for (int hrow = 0; hrow < y.rows(); hrow++) {
      dco::derivative(dco::value(y_active(hrow))) = 1; // wiggle y[hrow]
      L_active = lagrangian_value(t, x, y_active, p);  // compute h
      // set sensitivity to wobbles in h to 1
      dco::value(dco::derivative(L_active)) = 1;
      tape->interpret_adjoint_and_reset_to(start_position);
      for (int hcol = 0; hcol < y.rows(); hcol++) {
        d2Ldy2(hrow, hcol) = dco::derivative(dco::derivative(y_active[hcol]));
        // reset any accumulated values in
        dco::derivative(dco::derivative(y_active(hcol))) = 0;
        dco::value(dco::derivative(y_active(hcol))) = 0;
      }
      // no longer wiggling y[hrow]
      dco::derivative(dco::value(y_active(hrow))) = 0;
    }
    return d2Ldy2;
  }

  template <typename T, typename X, typename Y, int YDIMS_EXT, int PDIMS>
  auto norm_dLdy(T t, X x, Eigen::Vector<Y, YDIMS_EXT> const &y,
                 Eigen::Vector<T, PDIMS> const &p) const
      -> decltype(m_objective(t, x, y, p)) {
    using eltype = decltype(m_objective(t, x, y, p));
    Eigen::Vector<eltype, YDIMS_EXT> dLdY = grad_y_lagrangian(t, x, y, p);
    return dLdY.norm();
  }

  // someone needs to fix the return types in this file (not me)
  template <typename T, typename X, typename Y_ACTIVE_T, int YDIMS_EXT,
            int PDIMS>
  auto grad_y_norm_dLdy(T const t, X const x,
                        Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> const &y,
                        Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> {

    // is possible to write a reasonable driver for this, tbh
    using dco_tangent_t = typename dco::gt1s<Y_ACTIVE_T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();

    active_t L_active;
    Eigen::Vector<active_t, YDIMS_EXT> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    auto start_position = tape->get_position();
    Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> dLdy(y.rows());
    Eigen::Matrix<Y_ACTIVE_T, YDIMS_EXT, YDIMS_EXT> d2Ldy2(y.rows(), y.rows());
    for (int hrow = 0; hrow < y.rows(); hrow++) {
      dco::derivative(dco::value(y_active(hrow))) = 1; // wiggle y[hrow]
      L_active = lagrangian_value(t, x, y_active, p);  // compute h
      // set sensitivity to wobbles in h to 1
      dco::value(dco::derivative(L_active)) = 1;
      tape->interpret_adjoint_and_reset_to(start_position);
      dLdy(hrow) = dco::value(dco::derivative(L_active)); // harvest L^(2)
      for (int hcol = 0; hcol < y.rows(); hcol++) {
        // harvest y^(2)_(1)
        d2Ldy2(hrow, hcol) = dco::derivative(dco::derivative(y_active[hcol]));
        // reset any accumulated values
        dco::derivative(dco::derivative(y_active(hcol))) = 0;
        dco::value(dco::derivative(y_active(hcol))) = 0;
      }
      // no longer wiggling y[hrow]
      dco::derivative(dco::value(y_active(hrow))) = 0;
    }

    // Hessian is symmetric... I doubt this transpose has significant cost,
    // though.
    Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> res(y.rows());
    res = 2 * d2Ldy2.transpose() * dLdy;
    return res;
  }

  template <typename T, typename X, typename Y_ACTIVE_T, int YDIMS_EXT,
            int PDIMS>
  auto hess_y_norm_dLdy(T t, X x, Eigen::Vector<Y_ACTIVE_T, YDIMS_EXT> const &y,
                        Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Matrix<decltype(m_objective(t, x, y, p)), YDIMS_EXT,
                       YDIMS_EXT> {
    using dco_base_tangent_t = typename dco::gt1s<Y_ACTIVE_T>::type;
    using dco_tangent_t = typename dco::gt1s<dco_base_tangent_t>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();

    active_t L_active;
    Eigen::Vector<active_t, YDIMS_EXT> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    auto start_position = tape->get_position();

    Eigen::Matrix<Y_ACTIVE_T, YDIMS_EXT, YDIMS_EXT> res(y.size(), y.size());
    res = Eigen::Matrix<Y_ACTIVE_T, YDIMS_EXT, YDIMS_EXT>::Zero(y.size(),
                                                                y.size());
  }
};
#endif
