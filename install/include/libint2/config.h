/* include/libint2/config.h.  Generated from config.h.in by configure.  */
/*
 *  Copyright (C) 2004-2021 Edward F. Valeev
 *
 *  This file is part of Libint.
 *
 *  Libint is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Libint is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with Libint.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* This file is automatically processed by configure script.
   It MUST NOT be changed manually after configuration, otherwise
   the library will likely fail to compile or produce erroneous results
 */

#ifndef _libint2_include_libint2config_h_
#define _libint2_include_libint2config_h_

/* The host architecture. */
#define LIBINT_HOST_ARCH "x86_64-conda-linux-gnu"

/* The target architecture. */
#define LIBINT_TARGET_ARCH "x86_64-conda-linux-gnu"

/* The version number. */
#define LIBINT_VERSION "2.8.1"

/* The major version number. */
#define LIBINT_MAJOR_VERSION 2

/* The minor version number. */
#define LIBINT_MINOR_VERSION 8

/* The micro version number. */
#define LIBINT_MICRO_VERSION 1

/* Prefix for all names in API */
/* #undef LIBINT_API_PREFIX */

/* Max AM (same for all derivatives; if not defined see LIBINT_MAX_AM_LIST) */
#define LIBINT_MAX_AM 6

/* Max AM for integrals and their derivatives */
#define LIBINT_MAX_AM_LIST "6,5,4"

/* Max optimized AM (same for all derivatives; if not defined see LIBINT_OPT_AM_LIST) */
#define LIBINT_OPT_AM 4

/* Max optimized AM for integrals and their derivatives */
/* #undef LIBINT_OPT_AM_LIST */

/* Support 1-body derivatives up to this order */
#define INCLUDE_ONEBODY 2

/* Disable support for 1-body property derivatives */
#define DISABLE_ONEBODY_PROPERTY_DERIVS 1

/* Support ERI derivatives up to this order */
#define INCLUDE_ERI 2

/* Support 3-center ERI derivatives up to this order */
#define INCLUDE_ERI3 2

/* Support 2-center ERI derivatives up to this order */
#define INCLUDE_ERI2 2

/* Support G12 derivatives up to this order */
#define INCLUDE_G12 1

/* Support G12DKH derivatives up to this order */
/* #undef INCLUDE_G12DKH */

/* Max AM for one-body ints */
/* #undef ONEBODY_MAX_AM */

/* Max optimized AM for one-body ints */
/* #undef ONEBODY_OPT_AM */

/* Max order of spherical multipole ints */
#define MULTIPOLE_MAX_ORDER 10

/* Max AM for ERI (same for all derivatives; if not defined see ERI_MAX_AM_LIST) */
/* #undef ERI_MAX_AM */

/* Max AM for ERI and its derivatives */
#define ERI_MAX_AM_LIST "5,4,3"

/* Max optimized AM for ERI (same for all derivatives; if not defined see ERI_OPT_AM_LIST) */
/* #undef ERI_OPT_AM */

/* Max optimized AM for ERI and its derivatives */
/* #undef ERI_OPT_AM_LIST */

/* Max AM for 3-center ERI (same for all derivatives; if not defined see ERI3_MAX_AM_LIST) */
/* #undef ERI3_MAX_AM */

/* Max AM for 3-center ERI and its derivatives */
#define ERI3_MAX_AM_LIST "6,5,4"

/* Max optimized AM for 3-center ERI (same for all derivatives; if not defined see ERI3_OPT_AM_LIST) */
/* #undef ERI3_OPT_AM */

/* Max optimized AM for 3-center ERI and its derivatives */
/* #undef ERI3_OPT_AM_LIST */

/* If 1, assume will transform the "unpaired" center (e.g. a in (a|cd)) to solid harmonics */
/* #undef ERI3_PURE_SH */

/* Max AM for 2-center ERI (same for all derivatives; if not defined see ERI2_MAX_AM_LIST) */
/* #undef ERI2_MAX_AM */

/* Max AM for 2-center ERI and its derivatives */
#define ERI2_MAX_AM_LIST "6,5,4"

/* Max optimized AM for 2-center ERI (same for all derivatives; if not defined see ERI2_OPT_AM_LIST) */
/* #undef ERI2_OPT_AM */

/* Max optimized AM for 2-center ERI and its derivatives */
/* #undef ERI2_OPT_AM_LIST */

/* If 1, assume will transform to solid harmonics */
/* #undef ERI2_PURE_SH */

/* Max AM for G12 method integrals */
#define G12_MAX_AM 4

/* Max optimized AM for G12 method integrals */
#define G12_OPT_AM 4

/* Support [Ti,G12] ? */
#define SUPPORT_T1G12 1

/* Max AM for G12DKH method integrals */
/* #undef G12DKH_MAX_AM */

/* Max optimized AM for G12DKH method integrals */
/* #undef G12DKH_OPT_AM */

/* Whether integral sets can be unrolled */
#define LIBINT_ENABLE_UNROLLING 100

/* Whether generic code can be used */
/* #undef LIBINT_ENABLE_GENERIC_CODE */

/* maximum length of vectors */
/* #undef LIBINT_VECTOR_LENGTH */

/* how to vectorize */
/* #undef LIBINT_VECTOR_METHOD */

/* if can be controlled with posix_memalign, alignment size */

/* Specifies the ordering of cartesian Gaussians in a shell. Allowed values are defined at the bottom of this file -- also see configure.in */
#define LIBINT_CGSHELL_ORDERING 1

/* Specifies the ordering of solid harmonics Gaussians in a shell. Allowed values are defined at the bottom of this file -- also see configure.in */

/* Specifies the class of shell sets generated. Allowed values are defined at the bottom of this file -- also see configure.in */
#define LIBINT_SHELL_SET 1

/* User-defined real type */


/*Generate FMA instructions? */
/* #undef LIBINT_GENERATE_FMA */

/* Accumulate integrals to the buffer? */
/* #undef LIBINT_ACCUM_INTS */

/* Whether FLOP counting is supported */
/* #undef LIBINT_FLOP_COUNT */

/* Whether profile instrumentation will be enabled */
/* #undef LIBINT_PROFILE */

/* Support contracted integrals? */
#define LIBINT_CONTRACTED_INTS 1

/* Generate single evaluator type? */
#define LIBINT_SINGLE_EVALTYPE 1

/* Generate composite evaluators? */
#define LIBINT_USE_COMPOSITE_EVALUATORS 1

/* Strategy for ERI evaluation */
#define LIBINT_ERI_STRATEGY 1

/* --------------------------
  have C++ features?
   -------------------------- */
/* see lib/autoconf/ac_check_cpp11.m4 */

/* define if CXX compiler can compile C++11 */
#define LIBINT_HAS_CXX11 1

/* C++ compiler allows template with default params as template template parameter */
/* #undef CXX_ALLOWS_DEFPARAMTEMPLATE_AS_TEMPTEMPPARAM */

/* define if Eigen library is available. */

/* define if system-wide Boost.Preprocessor is available */

/*
  Known orderings of cartesian Gaussians
*/
#define LIBINT_CGSHELL_ORDERING_STANDARD 1
#define LIBINT_CGSHELL_ORDERING_INTV3 2
#define LIBINT_CGSHELL_ORDERING_GAMESS 3
#define LIBINT_CGSHELL_ORDERING_ORCA 4
#define LIBINT_CGSHELL_ORDERING_BAGEL 5

/*
  Known orderings of solid harmonic Gaussians
*/
#define LIBINT_SHGSHELL_ORDERING_STANDARD 1
#define LIBINT_SHGSHELL_ORDERING_GAUSSIAN 2

/*
  Known sets of shell sets
*/
#define LIBINT_SHELL_SET_STANDARD 1
#define LIBINT_SHELL_SET_ORCA 2

/*
 Libint-independent features
 */

/* have stdint.h ? */
#define HAVE_STDINT_H 1

/* have MPFR library ? */

/* have posix_memalign ? */

/* compiler type detection */
#define LIBINT_COMPILER_ID_GNU 0
#define LIBINT_COMPILER_ID_Clang 1
#define LIBINT_COMPILER_ID_AppleClang 2
#define LIBINT_COMPILER_ID_XLClang 3
#define LIBINT_COMPILER_ID_Intel 4
#if defined(__INTEL_COMPILER_BUILD_DATE)  /* macros like __ICC and even __INTEL_COMPILER can be affected by command options like -no-icc */
# define LIBINT_COMPILER_ID LIBINT_COMPILER_ID_Intel
# define LIBINT_COMPILER_IS_ICC 1
#endif
#if defined(__clang__) && !defined(LIBINT_COMPILER_IS_ICC)
# define LIBINT_COMPILER_IS_CLANG 1
# if defined(__apple_build_version__)
#  define LIBINT_COMPILER_ID LIBINT_COMPILER_ID_AppleClang
# elif defined(__ibmxl__)
#  define LIBINT_COMPILER_ID LIBINT_COMPILER_ID_XLClang
# else
#  define LIBINT_COMPILER_ID LIBINT_COMPILER_ID_Clang
# endif
#endif
#if defined(__GNUG__) && !defined(LIBINT_COMPILER_IS_ICC) && !defined(LIBINT_COMPILER_IS_CLANG)
# define LIBINT_COMPILER_ID LIBINT_COMPILER_ID_GNU
# define LIBINT_COMPILER_IS_GCC 1
#endif

/* ----------- pragma helpers ---------------*/
#define LIBINT_PRAGMA(x) _Pragma(#x)
/* same as LIBINT_PRAGMA(x), but expands x */
#define LIBINT_XPRAGMA(x) LIBINT_PRAGMA(x)
/* "concats" a and b with a space in between */
#define LIBINT_CONCAT(a,b) a b
#if defined(LIBINT_COMPILER_IS_CLANG)
#define LIBINT_PRAGMA_CLANG(x) LIBINT_XPRAGMA( LIBINT_CONCAT(clang,x) )
#else
#define LIBINT_PRAGMA_CLANG(x)
#endif
#if defined(LIBINT_COMPILER_IS_GCC)
#define LIBINT_PRAGMA_GCC(x) LIBINT_XPRAGMA( LIBINT_CONCAT(GCC,x) )
#else
#define LIBINT_PRAGMA_GCC(x)
#endif

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(deprecated)
#define LIBINT_DEPRECATED(msg) [[deprecated(msg)]]
#endif
#endif
#ifndef LIBINT_DEPRECATED
#define LIBINT_DEPRECATED(msg) LIBINT_XPRAGMA( LIBINT_CONCAT(message, msg) )
#endif

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(maybe_unused)
#define LIBINT_MAYBE_UNUSED [[maybe_unused]]
#endif
#endif  // __has_cpp_attribute
#ifndef LIBINT_MAYBE_UNUSED
#if defined __has_attribute
#  if __has_attribute (unused)
#    define LIBINT_MAYBE_UNUSED __attribute__ ((unused))
#  endif
#endif  // __has_attribute
#endif  // LIBINT_MAYBE_UNUSED
#ifndef LIBINT_MAYBE_UNUSED  // fallback
#define LIBINT_MAYBE_UNUSED
#endif

#endif /* header guard */
/* EXTRA DEFINES DETERMINED BY CONFIGURE OF THE EXPORTED LIBRARY */
#ifndef _libint2_include_libint2_config_h_1
#define _libint2_include_libint2_config_h_1
/* #undef LIBINT2_ALIGN_SIZE */
#define HAVE_POSIX_MEMALIGN 1
/* #undef LIBINT_HAS_MPFR */
#define LIBINT2_REALTYPE double
#define LIBINT_SHGSHELL_ORDERING 1
/* #undef LIBINT_HAS_EIGEN */
/* #undef LIBINT_HAS_SYSTEM_BOOST_PREPROCESSOR_VARIADICS */
#endif /* header guard #2 */
