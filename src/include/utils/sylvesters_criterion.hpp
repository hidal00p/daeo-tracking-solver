#ifndef _SYLVESTERS_CRITERION_HPP
#define _SYLVESTERS_CRITERION_HPP

#include <type_traits>

#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"
#include "eigen_interval_extensions.hpp"

/**
 * @brief Check if argument is less than or equal to zero.
 */
template <typename T> inline bool nonpositive(T arg) { return arg <= 0; }

/**
 * @brief Check if the lower end of the interval is less than or equal to zero.
 */
template <typename T, typename P>
inline bool nonpositive(boost::numeric::interval<T, P> arg) {
  return arg.lower() <= 0;
}

/**
 * @brief Check if the argument is greater than or equal to zero
 */
template <typename T> inline bool nonnegative(T arg) { return arg >= 0; }

/**
 * @brief Check if the upper end of the interval is greater than or equal to
 * zero
 */
template <typename T, typename P>
inline bool nonnegative(boost::numeric::interval<T, P> arg) {
  return arg.upper() >= 0;
}

struct drop_idx {
  Eigen::Index drop;
  Eigen::Index original_size;
  Eigen::Index size() const { return original_size - 1; }
  Eigen::Index operator[](Eigen::Index i) const {
    return (i < drop) ? i : i + 1;
  };
};

/**
 * @brief Inefficient determinant calculation.
 */
template <typename T, int N,
          std::enable_if_t<(N != Eigen::Dynamic), bool> = true>
typename Eigen::Matrix<T, N, N>::Scalar
bad_determinant(Eigen::Matrix<T, N, N> const &A) {
  using Eigen::seq;
  typename Eigen::Matrix<T, N, N>::Scalar det{0};
  if constexpr (N == 1) {
    det = A(0, 0);
  } else if constexpr (N == 2) {
    det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  } else if constexpr (N == 3) {
    det += A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1));
    det -= A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0));
    det += A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  } else {
    // det A = \sum_{i=1}^N -1^(i-1) * A(i,1)* det (i,1 submatrix of A)
    // even-numbered minors
    for (int i = 0; i < A.cols(); i += 2) {
      Eigen::Matrix<T, N - 1, N - 1> minor;
      minor = A(drop_idx{0, A.rows()}, drop_idx{i, A.cols()});
      det += A(0, i) * bad_determinant(minor);
    }
    // odd-numbered minors
    for (int i = 1; i < A.cols(); i += 2) {
      Eigen::Matrix<T, N - 1, N - 1> minor;
      minor = A(drop_idx{0, A.rows()}, drop_idx{i, A.cols()});
      det -= A(0, i) * bad_determinant(minor);
    }
  }
  return det;
}

template <typename T>
typename Eigen::MatrixX<T>::Scalar bad_determinant(Eigen::MatrixX<T> const &A) {
  using Eigen::seq;
  typename Eigen::MatrixX<T>::Scalar det{0};
  if (A.rows() == 1) {
    det = A(0, 0);
  } else if (A.rows() == 2) {
    det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  } else if (A.rows() == 3) {
    det += A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1));
    det -= A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0));
    det += A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  } else {
    // det A = \sum_{i=1}^N -1^(i-1) * A(i,1)* det (i,1 submatrix of A)
    // even-numbered minors
    for (int i = 0; i < A.cols(); i += 2) {
      Eigen::MatrixX<T> minor;
      minor = A(drop_idx{0, A.rows()}, drop_idx{i, A.cols()});
      det += A(0, i) * bad_determinant(minor);
      // det += A(drop_idx{0, A.rows()}, drop_idx{i, A.cols()}).rows();
    }
    // odd-numbered minors
    for (int i = 1; i < A.cols(); i += 2) {
      Eigen::MatrixX<T> minor;
      minor = A(drop_idx{0, A.rows()}, drop_idx{i, A.cols()});
      det -= A(0, i) * bad_determinant(minor);
    }
  }
  return det;
}

/**
 * @brief Test if each of the leading principal minors
 * (determinants of upper-left-justified square matrices)
 * of A are strictly positive.
 * @param[in] A
 * @param[in] skip The number of leading principal minors to skip testing
 * (default zero)
 */
template <typename T, int NDIMS>
bool leading_minors_positive(Eigen::Matrix<T, NDIMS, NDIMS> const &A,
                             Eigen::Index skip = 0) {
  for (int n = skip; n < A.rows(); n++) {
    Eigen::MatrixX<T> submatrix = A.block(0, 0, n + 1, n + 1);
    if (nonpositive(bad_determinant(submatrix))) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Test if each of the leading principal minors
 * (determinants of upper-left-justified square matrices)
 * of A are strictly negative.
 * @param[in] A
 * @param[in] skip The number of leading principal minors to skip testing
 * (default zero)
 */
template <typename T, int NDIMS>
bool leading_minors_negative(Eigen::Matrix<T, NDIMS, NDIMS> const &A,
                             Eigen::Index skip = 0) {
  for (int n = skip; n < A.rows(); n++) {
    Eigen::MatrixX<T> submatrix = A.block(0, 0, n + 1, n + 1);
    if (nonnegative(bad_determinant(submatrix))) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Test if each of the leading principal minors
 * (determinants of upper-left-justified square matrices)
 * of A alternate in sign, with minors of odd order < 0
 * and minors of even order > 0.
 * @param[in] A
 * @param[in] skip The number of leading principal minors to skip testing
 * (default zero)
 */
template <typename T, int NDIMS>
bool leading_minors_alternate(Eigen::Matrix<T, NDIMS, NDIMS> const &A,
                              Eigen::Index skip = 0) {
  for (int n = skip; n < A.rows(); n++) {
    Eigen::MatrixX<T> submatrix = A.block(0, 0, n + 1, n + 1);
    if ((submatrix.rows() & 1) && nonnegative(bad_determinant(submatrix))) {
      return false;
    } else if (nonpositive(bad_determinant(submatrix))) {
      return false;
    }
  }
  return true;
}

#endif