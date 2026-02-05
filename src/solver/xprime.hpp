#ifndef _XPRIME_FUNCTION_HPP
#define _XPRIME_FUNCTION_HPP

#include <type_traits>

#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"
#include "dco.hpp"

#include "utils/daeo_utils.hpp"

template <typename T> struct is_eigen_vec : std::false_type {};

template <typename T, int DIMS>
struct is_eigen_vec<Eigen::Vector<T, DIMS>> : std::true_type {};

template <typename T>
concept IsEigenVec = is_eigen_vec<T>::value;

template <typename FN, typename T, typename X, typename Y, int XDIMS, int YDIMS,
          int PDIMS>
concept YieldsEigenVec = requires(FN f, T t, Eigen::Vector<X, XDIMS> const &x,
                                  Eigen::Vector<Y, YDIMS> const &y,
                                  Eigen::Vector<T, PDIMS> const &p) {
  { f(t, x, y, p) } -> IsEigenVec;
};

template <typename FN> class WrappedXPrimeFunction {

  FN const m_fn;

  template <typename T, typename X, typename Y, int XDIMS, int YDIMS, int PDIMS>
  requires YieldsEigenVec<FN, T, X, Y, XDIMS, YDIMS, PDIMS>
  auto xprime_value(T t, Eigen::Vector<X, XDIMS> const &x,
                    Eigen::Vector<Y, YDIMS> const &y,
                    Eigen::Vector<T, PDIMS> const &p) const {
    return m_fn(t, x, y, p);
  }

  template <typename T, typename X, typename Y, int XDIMS, int YDIMS, int PDIMS>
  requires YieldsEigenVec<FN, T, X, Y, XDIMS, YDIMS, PDIMS>
  auto jacobian_x(T t, Eigen::Vector<X, XDIMS> const &x,
                  Eigen::Vector<Y, YDIMS> const &y,
                  Eigen::Vector<T, PDIMS> const &p) const
      -> Eigen::Matrix<typename decltype(m_fn(t, x, y, p))::Scalar, XDIMS,
                       XDIMS> {
    using dco_mode_t = dco::ga1s<X>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();
    Eigen::Vector<active_t, XDIMS> x_active(x.rows());
    for (int i = 0; i < x.rows(); i++) {
      dco::value(x_active(i)) = x(i);
      tape->register_variable(x_active(i));
    }
    auto start_position = tape->get_position();

    using res_t = decltype(m_fn(t, x, y, p));
    Eigen::Matrix<typename res_t::Scalar, XDIMS, XDIMS> jac_x(x.rows(),
                                                              x.rows());
    Eigen::Vector<active_t, XDIMS> f_active(x.rows());
    for (int i = 0; i < x.rows(); i++) {
      f_active = m_fn(t, x_active, y, p);
      dco::derivative(f_active(i)) = 1.0;
      tape->interpret_adjoint_and_reset_to(start_position);
      for (int j = 0; j < x.rows(); j++) {
        jac_x(i, j) = dco::derivative(x_active(j));
        dco::derivative(x_active(j)) = 0;
      }
    }
    return jac_x;
  }
};

#endif