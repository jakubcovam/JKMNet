// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"


// Test the corner cases of pow(x, y) for real types.
template<typename Scalar>
void pow_test() {
  const Scalar zero = Scalar(0);
  const Scalar eps = Eigen::NumTraits<Scalar>::epsilon();
  const Scalar one = Scalar(1);
  const Scalar two = Scalar(2);
  const Scalar three = Scalar(3);
  const Scalar sqrt_half = Scalar(std::sqrt(0.5));
  const Scalar sqrt2 = Scalar(std::sqrt(2));
  const Scalar inf = Eigen::NumTraits<Scalar>::infinity();
  const Scalar nan = Eigen::NumTraits<Scalar>::quiet_NaN();
  const Scalar denorm_min = EIGEN_ARCH_ARM ? zero : std::numeric_limits<Scalar>::denorm_min();
  const Scalar min = (std::numeric_limits<Scalar>::min)();
  const Scalar max = (std::numeric_limits<Scalar>::max)();
  const Scalar max_exp = (static_cast<Scalar>(int(Eigen::NumTraits<Scalar>::max_exponent())) * Scalar(EIGEN_LN2)) / eps;

  const static Scalar abs_vals[] = {zero,
                                    denorm_min,
                                    min,
                                    eps,
                                    sqrt_half,
                                    one,
                                    sqrt2,
                                    two,
                                    three,
                                    max_exp,
                                    max,
                                    inf,
                                    nan};
  const int abs_cases = 13;
  const int num_cases = 2*abs_cases * 2*abs_cases;
  // Repeat the same value to make sure we hit the vectorized path.
  const int num_repeats = 32;
  Array<Scalar, Dynamic, Dynamic> x(num_repeats, num_cases);
  Array<Scalar, Dynamic, Dynamic> y(num_repeats, num_cases);
  int count = 0;
  for (int i = 0; i < abs_cases; ++i) {
    const Scalar abs_x = abs_vals[i];
    for (int sign_x = 0; sign_x < 2; ++sign_x) {
      Scalar x_case = sign_x == 0 ? -abs_x : abs_x;
      for (int j = 0; j < abs_cases; ++j) {
        const Scalar abs_y = abs_vals[j];
        for (int sign_y = 0; sign_y < 2; ++sign_y) {
          Scalar y_case = sign_y == 0 ? -abs_y : abs_y;
          for (int repeat = 0; repeat < num_repeats; ++repeat) {
            x(repeat, count) = x_case;
            y(repeat, count) = y_case;
          }
          ++count;
        }
      }
    }
  }

  Array<Scalar, Dynamic, Dynamic> actual = x.pow(y);
  const Scalar tol = test_precision<Scalar>();
  bool all_pass = true;
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < num_cases; ++j) {
      Scalar e = static_cast<Scalar>(std::pow(x(i,j), y(i,j)));
      Scalar a = actual(i, j);
#if EIGEN_ARCH_ARM
      // Work around NEON flush-to-zero mode
      // if ref returns a subnormal value and Eigen returns 0, then skip the test
      if (a == Scalar(0) &&
          (e > -(std::numeric_limits<Scalar>::min)() && e < (std::numeric_limits<Scalar>::min)() &&
           e >= -std::numeric_limits<Scalar>::denorm_min() && e <= std::numeric_limits<Scalar>::denorm_min())) {
        continue;
      }
#endif
      bool success = (a == e) || ((numext::isfinite)(e) && internal::isApprox(a, e, tol)) ||
                     ((numext::isnan)(a) && (numext::isnan)(e));
      all_pass &= success;
      if (!success) {
        std::cout << "pow(" << x(i,j) << "," << y(i,j) << ")   =   " << a << " !=  " << e << std::endl;
      }
    }
  }
  VERIFY(all_pass);
}

template<typename ArrayType> void array(const ArrayType& m)
{
  typedef typename ArrayType::Scalar Scalar;
  typedef typename ArrayType::RealScalar RealScalar;
  typedef Array<Scalar, ArrayType::RowsAtCompileTime, 1> ColVectorType;
  typedef Array<Scalar, 1, ArrayType::ColsAtCompileTime> RowVectorType;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols),
             m2 = ArrayType::Random(rows, cols),
             m3(rows, cols);
  ArrayType m4 = m1; // copy constructor
  VERIFY_IS_APPROX(m1, m4);

  ColVectorType cv1 = ColVectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols);

  Scalar  s1 = internal::random<Scalar>(),
          s2 = internal::random<Scalar>();

  // scalar addition
  VERIFY_IS_APPROX(m1 + s1, s1 + m1);
  VERIFY_IS_APPROX(m1 + s1, ArrayType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX(s1 - m1, (-m1)+s1 );
  VERIFY_IS_APPROX(m1 - s1, m1 - ArrayType::Constant(rows,cols,s1));
  VERIFY_IS_APPROX(s1 - m1, ArrayType::Constant(rows,cols,s1) - m1);
  VERIFY_IS_APPROX((m1*Scalar(2)) - s2, (m1+m1) - ArrayType::Constant(rows,cols,s2) );
  m3 = m1;
  m3 += s2;
  VERIFY_IS_APPROX(m3, m1 + s2);
  m3 = m1;
  m3 -= s1;
  VERIFY_IS_APPROX(m3, m1 - s1);

  // scalar operators via Maps
  m3 = m1;
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) -= ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 - m2);

  m3 = m1;
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) += ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 + m2);

  m3 = m1;
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) *= ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 * m2);

  m3 = m1;
  m2 = ArrayType::Random(rows,cols);
  m2 = (m2==0).select(1,m2);
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) /= ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 / m2);

  // reductions
  VERIFY_IS_APPROX(m1.abs().colwise().sum().sum(), m1.abs().sum());
  VERIFY_IS_APPROX(m1.abs().rowwise().sum().sum(), m1.abs().sum());
  using std::abs;
  VERIFY_IS_MUCH_SMALLER_THAN(abs(m1.colwise().sum().sum() - m1.sum()), m1.abs().sum());
  VERIFY_IS_MUCH_SMALLER_THAN(abs(m1.rowwise().sum().sum() - m1.sum()), m1.abs().sum());
  if (!internal::isMuchSmallerThan(abs(m1.sum() - (m1+m2).sum()), m1.abs().sum(), test_precision<Scalar>()))
      VERIFY_IS_NOT_APPROX(((m1+m2).rowwise().sum()).sum(), m1.sum());
  VERIFY_IS_APPROX(m1.colwise().sum(), m1.colwise().redux(internal::scalar_sum_op<Scalar,Scalar>()));

  // vector-wise ops
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() += cv1, m1.colwise() + cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() -= cv1, m1.colwise() - cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() += rv1, m1.rowwise() + rv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() -= rv1, m1.rowwise() - rv1);

  // Conversion from scalar
  VERIFY_IS_APPROX((m3 = s1), ArrayType::Constant(rows,cols,s1));
  VERIFY_IS_APPROX((m3 = 1),  ArrayType::Constant(rows,cols,1));
  VERIFY_IS_APPROX((m3.topLeftCorner(rows,cols) = 1),  ArrayType::Constant(rows,cols,1));
  typedef Array<Scalar,
                ArrayType::RowsAtCompileTime==Dynamic?2:ArrayType::RowsAtCompileTime,
                ArrayType::ColsAtCompileTime==Dynamic?2:ArrayType::ColsAtCompileTime,
                ArrayType::Options> FixedArrayType;
  {
    FixedArrayType f1(s1);
    VERIFY_IS_APPROX(f1, FixedArrayType::Constant(s1));
    FixedArrayType f2(numext::real(s1));
    VERIFY_IS_APPROX(f2, FixedArrayType::Constant(numext::real(s1)));
    FixedArrayType f3((int)100*numext::real(s1));
    VERIFY_IS_APPROX(f3, FixedArrayType::Constant((int)100*numext::real(s1)));
    f1.setRandom();
    FixedArrayType f4(f1.data());
    VERIFY_IS_APPROX(f4, f1);
  }
  #if EIGEN_HAS_CXX11
  {
    FixedArrayType f1{s1};
    VERIFY_IS_APPROX(f1, FixedArrayType::Constant(s1));
    FixedArrayType f2{numext::real(s1)};
    VERIFY_IS_APPROX(f2, FixedArrayType::Constant(numext::real(s1)));
    FixedArrayType f3{(int)100*numext::real(s1)};
    VERIFY_IS_APPROX(f3, FixedArrayType::Constant((int)100*numext::real(s1)));
    f1.setRandom();
    FixedArrayType f4{f1.data()};
    VERIFY_IS_APPROX(f4, f1);
  }
  #endif

  // pow
  VERIFY_IS_APPROX(m1.pow(2), m1.square());
  VERIFY_IS_APPROX(pow(m1,2), m1.square());
  VERIFY_IS_APPROX(m1.pow(3), m1.cube());
  VERIFY_IS_APPROX(pow(m1,3), m1.cube());
  VERIFY_IS_APPROX((-m1).pow(3), -m1.cube());
  VERIFY_IS_APPROX(pow(2*m1,3), 8*m1.cube());
  ArrayType exponents = ArrayType::Constant(rows, cols, RealScalar(2));
  VERIFY_IS_APPROX(Eigen::pow(m1,exponents), m1.square());
  VERIFY_IS_APPROX(m1.pow(exponents), m1.square());
  VERIFY_IS_APPROX(Eigen::pow(2*m1,exponents), 4*m1.square());
  VERIFY_IS_APPROX((2*m1).pow(exponents), 4*m1.square());
  VERIFY_IS_APPROX(Eigen::pow(m1,2*exponents), m1.square().square());
  VERIFY_IS_APPROX(m1.pow(2*exponents), m1.square().square());
  VERIFY_IS_APPROX(Eigen::pow(m1(0,0), exponents), ArrayType::Constant(rows,cols,m1(0,0)*m1(0,0)));

  // Check possible conflicts with 1D ctor
  typedef Array<Scalar, Dynamic, 1> OneDArrayType;
  {
    OneDArrayType o1(rows);
    VERIFY(o1.size()==rows);
    OneDArrayType o2(static_cast<int>(rows));
    VERIFY(o2.size()==rows);
  }
  #if EIGEN_HAS_CXX11
  {
    OneDArrayType o1{rows};
    VERIFY(o1.size()==rows);
    OneDArrayType o4{int(rows)};
    VERIFY(o4.size()==rows);
  }
  #endif
  // Check possible conflicts with 2D ctor
  typedef Array<Scalar, Dynamic, Dynamic> TwoDArrayType;
  typedef Array<Scalar, 2, 1> ArrayType2;
  {
    TwoDArrayType o1(rows,cols);
    VERIFY(o1.rows()==rows);
    VERIFY(o1.cols()==cols);
    TwoDArrayType o2(static_cast<int>(rows),static_cast<int>(cols));
    VERIFY(o2.rows()==rows);
    VERIFY(o2.cols()==cols);

    ArrayType2 o3(rows,cols);
    VERIFY(o3(0)==Scalar(rows) && o3(1)==Scalar(cols));
    ArrayType2 o4(static_cast<int>(rows),static_cast<int>(cols));
    VERIFY(o4(0)==Scalar(rows) && o4(1)==Scalar(cols));
  }
  #if EIGEN_HAS_CXX11
  {
    TwoDArrayType o1{rows,cols};
    VERIFY(o1.rows()==rows);
    VERIFY(o1.cols()==cols);
    TwoDArrayType o2{int(rows),int(cols)};
    VERIFY(o2.rows()==rows);
    VERIFY(o2.cols()==cols);

    ArrayType2 o3{rows,cols};
    VERIFY(o3(0)==Scalar(rows) && o3(1)==Scalar(cols));
    ArrayType2 o4{int(rows),int(cols)};
    VERIFY(o4(0)==Scalar(rows) && o4(1)==Scalar(cols));
  }
  #endif
}

template<typename ArrayType> void comparisons(const ArrayType& m)
{
  using std::abs;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2 = ArrayType::Random(rows, cols),
            m3(rows, cols),
            m4 = m1;

  m4 = (m4.abs()==Scalar(0)).select(1,m4);

  VERIFY(((m1 + Scalar(1)) > m1).all());
  VERIFY(((m1 - Scalar(1)) < m1).all());
  if (rows*cols>1)
  {
    m3 = m1;
    m3(r,c) += 1;
    VERIFY(! (m1 < m3).all() );
    VERIFY(! (m1 > m3).all() );
  }
  VERIFY(!(m1 > m2 && m1 < m2).any());
  VERIFY((m1 <= m2 || m1 >= m2).all());

  // comparisons array to scalar
  VERIFY( (m1 != (m1(r,c)+1) ).any() );
  VERIFY( (m1 >  (m1(r,c)-1) ).any() );
  VERIFY( (m1 <  (m1(r,c)+1) ).any() );
  VERIFY( (m1 ==  m1(r,c)    ).any() );

  // comparisons scalar to array
  VERIFY( ( (m1(r,c)+1) != m1).any() );
  VERIFY( ( (m1(r,c)-1) <  m1).any() );
  VERIFY( ( (m1(r,c)+1) >  m1).any() );
  VERIFY( (  m1(r,c)    == m1).any() );

  // test Select
  VERIFY_IS_APPROX( (m1<m2).select(m1,m2), m1.cwiseMin(m2) );
  VERIFY_IS_APPROX( (m1>m2).select(m1,m2), m1.cwiseMax(m2) );
  Scalar mid = (m1.cwiseAbs().minCoeff() + m1.cwiseAbs().maxCoeff())/Scalar(2);
  for (int j=0; j<cols; ++j)
  for (int i=0; i<rows; ++i)
    m3(i,j) = abs(m1(i,j))<mid ? 0 : m1(i,j);
  VERIFY_IS_APPROX( (m1.abs()<ArrayType::Constant(rows,cols,mid))
                        .select(ArrayType::Zero(rows,cols),m1), m3);
  // shorter versions:
  VERIFY_IS_APPROX( (m1.abs()<ArrayType::Constant(rows,cols,mid))
                        .select(0,m1), m3);
  VERIFY_IS_APPROX( (m1.abs()>=ArrayType::Constant(rows,cols,mid))
                        .select(m1,0), m3);
  // even shorter version:
  VERIFY_IS_APPROX( (m1.abs()<mid).select(0,m1), m3);

  // count
  VERIFY(((m1.abs()+1)>RealScalar(0.1)).count() == rows*cols);

  // and/or
  VERIFY( (m1<RealScalar(0) && m1>RealScalar(0)).count() == 0);
  VERIFY( (m1<RealScalar(0) || m1>=RealScalar(0)).count() == rows*cols);
  RealScalar a = m1.abs().mean();
  VERIFY( (m1<-a || m1>a).count() == (m1.abs()>a).count());

  typedef Array<Index, Dynamic, 1> ArrayOfIndices;

  // TODO allows colwise/rowwise for array
  VERIFY_IS_APPROX(((m1.abs()+1)>RealScalar(0.1)).colwise().count(), ArrayOfIndices::Constant(cols,rows).transpose());
  VERIFY_IS_APPROX(((m1.abs()+1)>RealScalar(0.1)).rowwise().count(), ArrayOfIndices::Constant(rows, cols));
}

template<typename ArrayType> void array_real(const ArrayType& m)
{
  using std::abs;
  using std::sqrt;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2 = ArrayType::Random(rows, cols),
            m3(rows, cols),
            m4 = m1;

  // avoid denormalized values so verification doesn't fail on platforms that don't support them
  // denormalized behavior is tested elsewhere (unary_op_test, binary_ops_test)
  const Scalar min = (std::numeric_limits<Scalar>::min)();
  m1 = (m1.abs()<min).select(Scalar(0),m1);
  m2 = (m2.abs()<min).select(Scalar(0),m2);
  m4 = (m4.abs()<min).select(Scalar(1),m4);

  Scalar  s1 = internal::random<Scalar>();

  // these tests are mostly to check possible compilation issues with free-functions.
  VERIFY_IS_APPROX(m1.sin(), sin(m1));
  VERIFY_IS_APPROX(m1.cos(), cos(m1));
  VERIFY_IS_APPROX(m1.tan(), tan(m1));
  VERIFY_IS_APPROX(m1.asin(), asin(m1));
  VERIFY_IS_APPROX(m1.acos(), acos(m1));
  VERIFY_IS_APPROX(m1.atan(), atan(m1));
  VERIFY_IS_APPROX(m1.sinh(), sinh(m1));
  VERIFY_IS_APPROX(m1.cosh(), cosh(m1));
  VERIFY_IS_APPROX(m1.tanh(), tanh(m1));
#if EIGEN_HAS_CXX11_MATH
  VERIFY_IS_APPROX(m1.tanh().atanh(), atanh(tanh(m1)));
  VERIFY_IS_APPROX(m1.sinh().asinh(), asinh(sinh(m1)));
  VERIFY_IS_APPROX(m1.cosh().acosh(), acosh(cosh(m1)));
#endif
  VERIFY_IS_APPROX(m1.logistic(), logistic(m1));

  VERIFY_IS_APPROX(m1.arg(), arg(m1));
  VERIFY_IS_APPROX(m1.round(), round(m1));
  VERIFY_IS_APPROX(m1.rint(), rint(m1));
  VERIFY_IS_APPROX(m1.floor(), floor(m1));
  VERIFY_IS_APPROX(m1.ceil(), ceil(m1));
  VERIFY((m1.isNaN() == (Eigen::isnan)(m1)).all());
  VERIFY((m1.isInf() == (Eigen::isinf)(m1)).all());
  VERIFY((m1.isFinite() == (Eigen::isfinite)(m1)).all());
  VERIFY_IS_APPROX(m4.inverse(), inverse(m4));
  VERIFY_IS_APPROX(m1.abs(), abs(m1));
  VERIFY_IS_APPROX(m1.abs2(), abs2(m1));
  VERIFY_IS_APPROX(m1.square(), square(m1));
  VERIFY_IS_APPROX(m1.cube(), cube(m1));
  VERIFY_IS_APPROX(cos(m1+RealScalar(3)*m2), cos((m1+RealScalar(3)*m2).eval()));
  VERIFY_IS_APPROX(m1.sign(), sign(m1));
  VERIFY((m1.sqrt().sign().isNaN() == (Eigen::isnan)(sign(sqrt(m1)))).all());

  // avoid inf and NaNs so verification doesn't fail
  m3 = m4.abs();

  VERIFY_IS_APPROX(m3.sqrt(), sqrt(abs(m3)));
  VERIFY_IS_APPROX(m3.rsqrt(), Scalar(1)/sqrt(abs(m3)));
  VERIFY_IS_APPROX(rsqrt(m3), Scalar(1)/sqrt(abs(m3)));
  VERIFY_IS_APPROX(m3.log(), log(m3));
  VERIFY_IS_APPROX(m3.log1p(), log1p(m3));
  VERIFY_IS_APPROX(m3.log10(), log10(m3));
  VERIFY_IS_APPROX(m3.log2(), log2(m3));


  VERIFY((!(m1>m2) == (m1<=m2)).all());

  VERIFY_IS_APPROX(sin(m1.asin()), m1);
  VERIFY_IS_APPROX(cos(m1.acos()), m1);
  VERIFY_IS_APPROX(tan(m1.atan()), m1);
  VERIFY_IS_APPROX(sinh(m1), Scalar(0.5)*(exp(m1)-exp(-m1)));
  VERIFY_IS_APPROX(cosh(m1), Scalar(0.5)*(exp(m1)+exp(-m1)));
  VERIFY_IS_APPROX(tanh(m1), (Scalar(0.5)*(exp(m1)-exp(-m1)))/(Scalar(0.5)*(exp(m1)+exp(-m1))));
  VERIFY_IS_APPROX(logistic(m1), (Scalar(1)/(Scalar(1)+exp(-m1))));
  VERIFY_IS_APPROX(arg(m1), ((m1<Scalar(0)).template cast<Scalar>())*Scalar(std::acos(Scalar(-1))));
  VERIFY((round(m1) <= ceil(m1) && round(m1) >= floor(m1)).all());
  VERIFY((rint(m1) <= ceil(m1) && rint(m1) >= floor(m1)).all());
  VERIFY(((ceil(m1) - round(m1)) <= Scalar(0.5) || (round(m1) - floor(m1)) <= Scalar(0.5)).all());
  VERIFY(((ceil(m1) - round(m1)) <= Scalar(1.0) && (round(m1) - floor(m1)) <= Scalar(1.0)).all());
  VERIFY(((ceil(m1) - rint(m1)) <= Scalar(0.5) || (rint(m1) - floor(m1)) <= Scalar(0.5)).all());
  VERIFY(((ceil(m1) - rint(m1)) <= Scalar(1.0) && (rint(m1) - floor(m1)) <= Scalar(1.0)).all());
  VERIFY((Eigen::isnan)((m1*Scalar(0))/Scalar(0)).all());
  VERIFY((Eigen::isinf)(m4/Scalar(0)).all());
  VERIFY(((Eigen::isfinite)(m1) && (!(Eigen::isfinite)(m1*Scalar(0)/Scalar(0))) && (!(Eigen::isfinite)(m4/Scalar(0)))).all());
  VERIFY_IS_APPROX(inverse(inverse(m4)),m4);
  VERIFY((abs(m1) == m1 || abs(m1) == -m1).all());
  VERIFY_IS_APPROX(m3, sqrt(abs2(m3)));
  VERIFY_IS_APPROX(m1.absolute_difference(m2), (m1 > m2).select(m1 - m2, m2 - m1));
  VERIFY_IS_APPROX( m1.sign(), -(-m1).sign() );
  VERIFY_IS_APPROX( m1*m1.sign(),m1.abs());
  VERIFY_IS_APPROX(m1.sign() * m1.abs(), m1);

  VERIFY_IS_APPROX(numext::abs2(numext::real(m1)) + numext::abs2(numext::imag(m1)), numext::abs2(m1));
  VERIFY_IS_APPROX(numext::abs2(Eigen::real(m1)) + numext::abs2(Eigen::imag(m1)), numext::abs2(m1));
  if(!NumTraits<Scalar>::IsComplex)
    VERIFY_IS_APPROX(numext::real(m1), m1);

  // shift argument of logarithm so that it is not zero
  Scalar smallNumber = NumTraits<Scalar>::dummy_precision();
  VERIFY_IS_APPROX((m3 + smallNumber).log() , log(abs(m3) + smallNumber));
  VERIFY_IS_APPROX((m3 + smallNumber + Scalar(1)).log() , log1p(abs(m3) + smallNumber));

  VERIFY_IS_APPROX(m1.exp() * m2.exp(), exp(m1+m2));
  VERIFY_IS_APPROX(m1.exp(), exp(m1));
  VERIFY_IS_APPROX(m1.exp() / m2.exp(),(m1-m2).exp());

  VERIFY_IS_APPROX(m1.expm1(), expm1(m1));
  VERIFY_IS_APPROX((m3 + smallNumber).exp() - Scalar(1), expm1(abs(m3) + smallNumber));

  VERIFY_IS_APPROX(m3.pow(RealScalar(0.5)), m3.sqrt());
  VERIFY_IS_APPROX(pow(m3,RealScalar(0.5)), m3.sqrt());

  VERIFY_IS_APPROX(m3.pow(RealScalar(-0.5)), m3.rsqrt());
  VERIFY_IS_APPROX(pow(m3,RealScalar(-0.5)), m3.rsqrt());

  // Avoid inf and NaN.
  m3 = (m1.square()<NumTraits<Scalar>::epsilon()).select(Scalar(1),m3);
  VERIFY_IS_APPROX(m3.pow(RealScalar(-2)), m3.square().inverse());
  pow_test<Scalar>();

  VERIFY_IS_APPROX(log10(m3), log(m3)/numext::log(Scalar(10)));
  VERIFY_IS_APPROX(log2(m3), log(m3)/numext::log(Scalar(2)));

  // scalar by array division
  const RealScalar tiny = sqrt(std::numeric_limits<RealScalar>::epsilon());
  s1 += Scalar(tiny);
  m1 += ArrayType::Constant(rows,cols,Scalar(tiny));
  VERIFY_IS_APPROX(s1/m1, s1 * m1.inverse());

  // check inplace transpose
  m3 = m1;
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3, m1.transpose());
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3, m1);
}

template<typename ArrayType> void array_complex(const ArrayType& m)
{
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2(rows, cols),
            m4 = m1;

  m4.real() = (m4.real().abs()==RealScalar(0)).select(RealScalar(1),m4.real());
  m4.imag() = (m4.imag().abs()==RealScalar(0)).select(RealScalar(1),m4.imag());

  Array<RealScalar, -1, -1> m3(rows, cols);

  for (Index i = 0; i < m.rows(); ++i)
    for (Index j = 0; j < m.cols(); ++j)
      m2(i,j) = sqrt(m1(i,j));

  // these tests are mostly to check possible compilation issues with free-functions.
  VERIFY_IS_APPROX(m1.sin(), sin(m1));
  VERIFY_IS_APPROX(m1.cos(), cos(m1));
  VERIFY_IS_APPROX(m1.tan(), tan(m1));
  VERIFY_IS_APPROX(m1.sinh(), sinh(m1));
  VERIFY_IS_APPROX(m1.cosh(), cosh(m1));
  VERIFY_IS_APPROX(m1.tanh(), tanh(m1));
  VERIFY_IS_APPROX(m1.logistic(), logistic(m1));
  VERIFY_IS_APPROX(m1.arg(), arg(m1));
  VERIFY((m1.isNaN() == (Eigen::isnan)(m1)).all());
  VERIFY((m1.isInf() == (Eigen::isinf)(m1)).all());
  VERIFY((m1.isFinite() == (Eigen::isfinite)(m1)).all());
  VERIFY_IS_APPROX(m4.inverse(), inverse(m4));
  VERIFY_IS_APPROX(m1.log(), log(m1));
  VERIFY_IS_APPROX(m1.log10(), log10(m1));
  VERIFY_IS_APPROX(m1.log2(), log2(m1));
  VERIFY_IS_APPROX(m1.abs(), abs(m1));
  VERIFY_IS_APPROX(m1.abs2(), abs2(m1));
  VERIFY_IS_APPROX(m1.sqrt(), sqrt(m1));
  VERIFY_IS_APPROX(m1.square(), square(m1));
  VERIFY_IS_APPROX(m1.cube(), cube(m1));
  VERIFY_IS_APPROX(cos(m1+RealScalar(3)*m2), cos((m1+RealScalar(3)*m2).eval()));
  VERIFY_IS_APPROX(m1.sign(), sign(m1));


  VERIFY_IS_APPROX(m1.exp() * m2.exp(), exp(m1+m2));
  VERIFY_IS_APPROX(m1.exp(), exp(m1));
  VERIFY_IS_APPROX(m1.exp() / m2.exp(),(m1-m2).exp());

  VERIFY_IS_APPROX(m1.expm1(), expm1(m1));
  VERIFY_IS_APPROX(expm1(m1), exp(m1) - 1.);
  // Check for larger magnitude complex numbers that expm1 matches exp - 1.
  VERIFY_IS_APPROX(expm1(10. * m1), exp(10. * m1) - 1.);

  VERIFY_IS_APPROX(sinh(m1), 0.5*(exp(m1)-exp(-m1)));
  VERIFY_IS_APPROX(cosh(m1), 0.5*(exp(m1)+exp(-m1)));
  VERIFY_IS_APPROX(tanh(m1), (0.5*(exp(m1)-exp(-m1)))/(0.5*(exp(m1)+exp(-m1))));
  VERIFY_IS_APPROX(logistic(m1), (1.0/(1.0 + exp(-m1))));

  for (Index i = 0; i < m.rows(); ++i)
    for (Index j = 0; j < m.cols(); ++j)
      m3(i,j) = std::atan2(m1(i,j).imag(), m1(i,j).real());
  VERIFY_IS_APPROX(arg(m1), m3);

  std::complex<RealScalar> zero(0.0,0.0);
  VERIFY((Eigen::isnan)(m1*zero/zero).all());
#if EIGEN_COMP_MSVC
  // msvc complex division is not robust
  VERIFY((Eigen::isinf)(m4/RealScalar(0)).all());
#else
#if EIGEN_COMP_CLANG
  // clang's complex division is notoriously broken too
  if((numext::isinf)(m4(0,0)/RealScalar(0))) {
#endif
    VERIFY((Eigen::isinf)(m4/zero).all());
#if EIGEN_COMP_CLANG
  }
  else
  {
    VERIFY((Eigen::isinf)(m4.real()/zero.real()).all());
  }
#endif
#endif // MSVC

  VERIFY(((Eigen::isfinite)(m1) && (!(Eigen::isfinite)(m1*zero/zero)) && (!(Eigen::isfinite)(m1/zero))).all());

  VERIFY_IS_APPROX(inverse(inverse(m4)),m4);
  VERIFY_IS_APPROX(conj(m1.conjugate()), m1);
  VERIFY_IS_APPROX(abs(m1), sqrt(square(m1.real())+square(m1.imag())));
  VERIFY_IS_APPROX(abs(m1), sqrt(abs2(m1)));
  VERIFY_IS_APPROX(log10(m1), log(m1)/log(10));
  VERIFY_IS_APPROX(log2(m1), log(m1)/log(2));

  VERIFY_IS_APPROX( m1.sign(), -(-m1).sign() );
  VERIFY_IS_APPROX( m1.sign() * m1.abs(), m1);

  // scalar by array division
  Scalar  s1 = internal::random<Scalar>();
  const RealScalar tiny = std::sqrt(std::numeric_limits<RealScalar>::epsilon());
  s1 += Scalar(tiny);
  m1 += ArrayType::Constant(rows,cols,Scalar(tiny));
  VERIFY_IS_APPROX(s1/m1, s1 * m1.inverse());

  // check inplace transpose
  m2 = m1;
  m2.transposeInPlace();
  VERIFY_IS_APPROX(m2, m1.transpose());
  m2.transposeInPlace();
  VERIFY_IS_APPROX(m2, m1);
  // Check vectorized inplace transpose.
  ArrayType m5 = ArrayType::Random(131, 131);
  ArrayType m6 = m5;
  m6.transposeInPlace();
  VERIFY_IS_APPROX(m6, m5.transpose());
}

template<typename ArrayType> void min_max(const ArrayType& m)
{
  typedef typename ArrayType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols);

  // min/max with array
  Scalar maxM1 = m1.maxCoeff();
  Scalar minM1 = m1.minCoeff();

  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, minM1), (m1.min)(ArrayType::Constant(rows,cols, minM1)));
  VERIFY_IS_APPROX(m1, (m1.min)(ArrayType::Constant(rows,cols, maxM1)));

  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, maxM1), (m1.max)(ArrayType::Constant(rows,cols, maxM1)));
  VERIFY_IS_APPROX(m1, (m1.max)(ArrayType::Constant(rows,cols, minM1)));

  // min/max with scalar input
  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, minM1), (m1.min)( minM1));
  VERIFY_IS_APPROX(m1, (m1.min)( maxM1));

  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, maxM1), (m1.max)( maxM1));
  VERIFY_IS_APPROX(m1, (m1.max)( minM1));


  // min/max with various NaN propagation options.
  if (m1.size() > 1 && !NumTraits<Scalar>::IsInteger) {
    m1(0,0) = NumTraits<Scalar>::quiet_NaN();
    maxM1 = m1.template maxCoeff<PropagateNaN>();
    minM1 = m1.template minCoeff<PropagateNaN>();
    VERIFY((numext::isnan)(maxM1));
    VERIFY((numext::isnan)(minM1));

    maxM1 = m1.template maxCoeff<PropagateNumbers>();
    minM1 = m1.template minCoeff<PropagateNumbers>();
    VERIFY(!(numext::isnan)(maxM1));
    VERIFY(!(numext::isnan)(minM1));
  }
}

template<int N>
struct shift_left {
  template<typename Scalar>
  Scalar operator()(const Scalar& v) const {
    return v << N;
  }
};

template<int N>
struct arithmetic_shift_right {
  template<typename Scalar>
  Scalar operator()(const Scalar& v) const {
    return v >> N;
  }
};

template<typename ArrayType> void array_integer(const ArrayType& m)
{
  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2(rows, cols);

  m2 = m1.template shiftLeft<2>();
  VERIFY( (m2 == m1.unaryExpr(shift_left<2>())).all() );
  m2 = m1.template shiftLeft<9>();
  VERIFY( (m2 == m1.unaryExpr(shift_left<9>())).all() );
  
  m2 = m1.template shiftRight<2>();
  VERIFY( (m2 == m1.unaryExpr(arithmetic_shift_right<2>())).all() );
  m2 = m1.template shiftRight<9>();
  VERIFY( (m2 == m1.unaryExpr(arithmetic_shift_right<9>())).all() );
}

EIGEN_DECLARE_TEST(array_cwise)
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( array(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( array(Array22f()) );
    CALL_SUBTEST_3( array(Array44d()) );
    CALL_SUBTEST_4( array(ArrayXXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_5( array(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( array(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( array(Array<Index,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( array_integer(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( array_integer(Array<Index,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( comparisons(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( comparisons(Array22f()) );
    CALL_SUBTEST_3( comparisons(Array44d()) );
    CALL_SUBTEST_5( comparisons(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( comparisons(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( min_max(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( min_max(Array22f()) );
    CALL_SUBTEST_3( min_max(Array44d()) );
    CALL_SUBTEST_5( min_max(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( min_max(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( array_real(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( array_real(Array22f()) );
    CALL_SUBTEST_3( array_real(Array44d()) );
    CALL_SUBTEST_5( array_real(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_7( array_real(Array<Eigen::half, 32, 32>()) );
    CALL_SUBTEST_8( array_real(Array<Eigen::bfloat16, 32, 32>()) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_4( array_complex(ArrayXXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }

  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<int>::type, int >::value));
  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<float>::type, float >::value));
  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<Array2i>::type, ArrayBase<Array2i> >::value));
  typedef CwiseUnaryOp<internal::scalar_abs_op<double>, ArrayXd > Xpr;
  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<Xpr>::type,
                           ArrayBase<Xpr>
                         >::value));
}
