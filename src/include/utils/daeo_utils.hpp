#ifndef _DAEO_UTILS_HPP
#define _DAEO_UTILS_HPP

#include <type_traits>

/**
 * Struct to propagate Eigen::Dynamic size Eigen types at compile time.
 */

#include "boost/numeric/interval.hpp"
// Go on an adventure to find Eigen::Dynamic
// Should be here (Eigen 3.4)
#include "Eigen/Core"
#include "fmt/core.h"
#include "fmt/format.h"

/**
 * @brief If @c DIMS_IN is the same as @c Eigen::Dynamic , @c value is equal to
 * @c Eigen::Dynamic . Otherwise, @c value is equal to @c DIMS_OUT
 */
template <int DIMS_IN, int DIMS_OUT>
struct propagate_dynamic : std::integral_constant<int, DIMS_OUT> {};

template <int DIMS_OUT>
struct propagate_dynamic<Eigen::Dynamic, DIMS_OUT>
    : std::integral_constant<int, Eigen::Dynamic> {};

template <int DIMS_IN, int DIMS_OUT>
inline constexpr int propagate_dynamic_v =
    propagate_dynamic<DIMS_IN, DIMS_OUT>::value;

/**
 * NTuple type, for the lazy programmer in me.
 */
template <std::size_t N, typename T> struct ntuple_detail {
  template <typename... Ts>
  using type = typename ntuple_detail<N - 1, T>::template type<T, Ts...>;
};

template <typename T> struct ntuple_detail<0, T> {
  template <typename... Ts> using type = typename std::tuple<Ts...>;
};

/**
 * @brief Tuple of `N` items of type `T`.
 */
template <std::size_t N, typename T>
using ntuple = typename ntuple_detail<N, T>::template type<>;

/**
 * Allow fmt to format boost intervals neatly :)
 */

template <typename T, typename P>
struct fmt::formatter<boost::numeric::interval<T, P>>
    : public fmt::formatter<T> {

  template <typename FormatContext>
  auto format(boost::numeric::interval<T, P> const &ival,
              FormatContext &ctx) const {
    auto &&out = ctx.out();
    fmt::format_to(out, "[");
    fmt::formatter<T>::format(ival.lower(), ctx);
    fmt::format_to(out, ", ");
    fmt::formatter<T>::format(ival.upper(), ctx);
    return fmt::format_to(out, "]");
  }
};

/**
 * @brief Check of an interval @c y contains zero, or if both its upper and
 * lower bounds are absolutely close to zero.
 */
template <typename T, typename POLICIES>
bool zero_in_or_absolutely_near(boost::numeric::interval<T, POLICIES> y,
                                T tol) {
  return boost::numeric::zero_in(y) ||
         (fabs(y.lower()) < tol && fabs(y.upper()) < tol);
}

/**
 * @brief Suggested Boost interval policies.
 */
template <typename T>
using suggested_interval_policies = boost::numeric::interval_lib::policies<
    boost::numeric::interval_lib::save_state<
        boost::numeric::interval_lib::rounded_transc_std<T>>,
    boost::numeric::interval_lib::checking_base<T>>;

#endif