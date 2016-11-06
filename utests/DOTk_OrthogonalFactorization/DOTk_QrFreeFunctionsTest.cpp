/*
 * DOTk_QrFreeFunctionsTest.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_QR.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_ColumnMatrix.hpp"
#include "DOTk_ColumnMatrix.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_UpperTriangularMatrix.hpp"
#include "DOTk_UpperTriangularMatrix.cpp"

namespace DOTkQrFreeFunctionsTest
{

template<typename Type>
inline void checkOrthogonality(const size_t & basis_dim_, dotk::matrix<Type> & matrix_, Type tolerance_ = 1e-8)
{
    for(size_t i = 0; i < basis_dim_; ++i)
    {
        for(size_t j = 0; j < basis_dim_; ++j)
        {
            if(i != j)
            {
                Type value = matrix_.basis(i)->dot(*matrix_.basis(j));
                EXPECT_NEAR(0., value, tolerance_);
            }
        }
    }
}

TEST(OrthoMethod, classicalGramSchmidtE1)
{
    size_t ncols = 4;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> Q(x, nrows);
    dotk::serial::DOTk_UpperTriangularMatrix<Real> R(nrows);
    // basis 1
    Q(0,0) = 1.;
    Q(0,1) = 1.;
    Q(0,2) = 1.;
    Q(0,3) = 1.;
    // basis 2
    Q(1,0) = -1.;
    Q(1,1) = 4.;
    Q(1,2) = 4.;
    Q(1,3) = 1.;
    // basis 3
    Q(2,0) = 4.;
    Q(2,1) = -2.;
    Q(2,2) = 2.;
    Q(2,3) = 0.;

    dotk::qr::classicalGramSchmidt(Q, R);

    std::vector<Real> gold(ncols, 0.);
    gold[0] = 0.5;
    gold[1] = 0.5;
    gold[2] = 0.5;
    gold[3] = 0.5;
    dotk::gtest::checkResults(ncols, gold.data(), *Q.basis(0));
    gold[0] = -0.707106781186548;
    gold[1] = 0.471404520791032;
    gold[2] = 0.471404520791032;
    gold[3] = -0.235702260395516;
    dotk::gtest::checkResults(ncols, gold.data(), *Q.basis(1));
    gold[0] = 0.288675134594813;
    gold[1] = -0.481125224324688;
    gold[2] = 0.673575314054564;
    gold[3] = -0.481125224324688;
    dotk::gtest::checkResults(ncols, gold.data(), *Q.basis(2));

    Real gold_R[] = {2, 4, 2, 4.24264068711928, -2.82842712474619, 3.46410161513775};
    std::vector<Real> data(R.size());
    R.gather(data.size(), data.data());
    dotk::gtest::checkResults(6, gold_R, data.size(), data.data());
}

TEST(OrthoMethod, classicalGramSchmidtE2)
{
    size_t nrows = 4;
    size_t ncols = 3;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> Q(x, ncols);
    dotk::serial::DOTk_UpperTriangularMatrix<Real> R(ncols);
    // basis 1
    Q(0,0) = 1.;
    Q(1,0) = 1.;
    Q(2,0) = 1.;
    Q(3,0) = 1.;
    // basis 2
    Q(0,1) = -1.;
    Q(1,1) = 4.;
    Q(2,1) = 4.;
    Q(3,1) = -1.;
    // basis 3
    Q(0,2) = 4.;
    Q(1,2) = -2.;
    Q(2,2) = 2.;
    Q(3,2) = 0.;

    dotk::qr::classicalGramSchmidt(Q, R);

    std::vector<Real> gold_Q(nrows, 0.);
    gold_Q[0] = 0.5;
    gold_Q[1] = 0.5;
    gold_Q[2] = 0.5;
    gold_Q[3] = 0.5;
    dotk::gtest::checkResults(nrows, gold_Q.data(), *Q.basis(0));
    gold_Q[0] = -0.5;
    gold_Q[1] = 0.5;
    gold_Q[2] = 0.5;
    gold_Q[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold_Q.data(), *Q.basis(1));
    gold_Q[0] = 0.5;
    gold_Q[1] = -0.5;
    gold_Q[2] = 0.5;
    gold_Q[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold_Q.data(), *Q.basis(2));

    Real gold_R[] = {2., 3., 2., 5., -2., 4.};
    std::vector<Real> data(R.size());
    R.gather(data.size(), data.data());
    dotk::gtest::checkResults(6, gold_R, data.size(), data.data());
}

TEST(OrthoMethod, classicalGramSchmidtE3)
{
    size_t nrows = 4;
    size_t ncols = 3;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> A(x, ncols);
    dotk::serial::DOTk_ColumnMatrix<Real> Q(x, ncols);
    dotk::serial::DOTk_UpperTriangularMatrix<Real> R(ncols);
    // basis 1
    A(0,0) = 1.;
    A(1,0) = 1.;
    A(2,0) = 1.;
    A(3,0) = 1.;
    // basis 2
    A(0,1) = -1.;
    A(1,1) = 4.;
    A(2,1) = 4.;
    A(3,1) = -1.;
    // basis 3
    A(0,2) = 4.;
    A(1,2) = -2.;
    A(2,2) = 2.;
    A(3,2) = 0.;

    dotk::qr::classicalGramSchmidt(A, Q, R);

    // check A values
    std::vector<Real> gold(nrows, 0.);
    gold[0] = 1.;
    gold[1] = 1.;
    gold[2] = 1.;
    gold[3] = 1.;
    dotk::gtest::checkResults(nrows, gold.data(), *A.basis(0));
    gold[0] = -1.;
    gold[1] = 4.;
    gold[2] = 4.;
    gold[3] = -1.;
    dotk::gtest::checkResults(nrows, gold.data(), *A.basis(1));
    gold[0] = 4.;
    gold[1] = -2.;
    gold[2] = 2.;
    gold[3] = 0.;
    dotk::gtest::checkResults(nrows, gold.data(), *A.basis(2));

    // check Q values
    gold[0] = 0.5;
    gold[1] = 0.5;
    gold[2] = 0.5;
    gold[3] = 0.5;
    dotk::gtest::checkResults(nrows, gold.data(), *Q.basis(0));
    gold[0] = -0.5;
    gold[1] = 0.5;
    gold[2] = 0.5;
    gold[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold.data(), *Q.basis(1));
    gold[0] = 0.5;
    gold[1] = -0.5;
    gold[2] = 0.5;
    gold[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold.data(), *Q.basis(2));

    // check R values
    Real gold_R[] = {2., 3., 2., 5., -2., 4.};
    std::vector<Real> data(R.size());
    R.gather(data.size(), data.data());
    dotk::gtest::checkResults(6, gold_R, data.size(), data.data());
}

TEST(OrthoMethod, modifiedGramSchmidtE1)
{
    size_t ncols = 4;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> Q(x, nrows);
    dotk::serial::DOTk_UpperTriangularMatrix<Real> R(nrows);
    // basis 1
    Q(0,0) = 2.;
    Q(0,1) = 0.;
    Q(0,2) = 0.;
    Q(0,3) = 0.;
    // basis 2
    Q(1,0) = 1.;
    Q(1,1) = 1.;
    Q(1,2) = 1.;
    Q(1,3) = 1.;
    // basis 3
    Q(2,0) = 2.;
    Q(2,1) = 0.;
    Q(2,2) = 2.;
    Q(2,3) = 0.;

    dotk::qr::modifiedGramSchmidt(Q, R);

    std::vector<Real> gold_Q(ncols, 0.);
    gold_Q[0] = 1.;
    gold_Q[1] = 0.;
    gold_Q[2] = 0.;
    gold_Q[3] = 0.;
    dotk::gtest::checkResults(ncols, gold_Q.data(), *Q.basis(0));
    gold_Q[0] = 0.;
    gold_Q[1] = 0.577350269189626;
    gold_Q[2] = 0.577350269189626;
    gold_Q[3] = 0.577350269189626;
    dotk::gtest::checkResults(ncols, gold_Q.data(), *Q.basis(1));
    gold_Q[0] = 0.;
    gold_Q[1] = -0.408248290463863;
    gold_Q[2] = 0.816496580927726;
    gold_Q[3] = -0.408248290463863;
    dotk::gtest::checkResults(ncols, gold_Q.data(), *Q.basis(2));

    Real gold_R[] = {2, 1, 2, 1.73205080756888, 1.15470053837925, 1.63299316185545};
    std::vector<Real> data(R.size());
    R.gather(data.size(), data.data());
    dotk::gtest::checkResults(6, gold_R, data.size(), data.data());
}

TEST(OrthoMethod, modifiedGramSchmidtE2)
{
    size_t nrows = 4;
    size_t ncols = 3;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> Q(x, ncols);
    dotk::serial::DOTk_UpperTriangularMatrix<Real> R(ncols);
    // basis 1
    Q(0,0) = 1.;
    Q(1,0) = 1.;
    Q(2,0) = 1.;
    Q(3,0) = 1.;
    // basis 2
    Q(0,1) = -1.;
    Q(1,1) = 4.;
    Q(2,1) = 4.;
    Q(3,1) = -1.;
    // basis 3
    Q(0,2) = 4.;
    Q(1,2) = -2.;
    Q(2,2) = 2.;
    Q(3,2) = 0.;

    dotk::qr::modifiedGramSchmidt(Q, R);

    std::vector<Real> gold_Q(nrows, 0.);
    gold_Q[0] = 0.5;
    gold_Q[1] = 0.5;
    gold_Q[2] = 0.5;
    gold_Q[3] = 0.5;
    dotk::gtest::checkResults(nrows, gold_Q.data(), *Q.basis(0));
    gold_Q[0] = -0.5;
    gold_Q[1] = 0.5;
    gold_Q[2] = 0.5;
    gold_Q[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold_Q.data(), *Q.basis(1));
    gold_Q[0] = 0.5;
    gold_Q[1] = -0.5;
    gold_Q[2] = 0.5;
    gold_Q[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold_Q.data(), *Q.basis(2));

    Real gold_R[] = {2., 3., 2., 5., -2., 4.};
    std::vector<Real> data(R.size());
    R.gather(data.size(), data.data());
    dotk::gtest::checkResults(6, gold_R, data.size(), data.data());
}

TEST(OrthoMethod, modifiedGramSchmidtE3)
{
    size_t nrows = 4;
    size_t ncols = 3;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> A(x, ncols);
    dotk::serial::DOTk_ColumnMatrix<Real> Q(x, ncols);
    dotk::serial::DOTk_UpperTriangularMatrix<Real> R(ncols);
    // basis 1
    A(0,0) = 1.;
    A(1,0) = 1.;
    A(2,0) = 1.;
    A(3,0) = 1.;
    // basis 2
    A(0,1) = -1.;
    A(1,1) = 4.;
    A(2,1) = 4.;
    A(3,1) = -1.;
    // basis 3
    A(0,2) = 4.;
    A(1,2) = -2.;
    A(2,2) = 2.;
    A(3,2) = 0.;

    dotk::qr::modifiedGramSchmidt(A, Q, R);

    // check A values
    std::vector<Real> gold(nrows, 0.);
    gold[0] = 1.;
    gold[1] = 1.;
    gold[2] = 1.;
    gold[3] = 1.;
    dotk::gtest::checkResults(nrows, gold.data(), *A.basis(0));
    gold[0] = -1.;
    gold[1] = 4.;
    gold[2] = 4.;
    gold[3] = -1.;
    dotk::gtest::checkResults(nrows, gold.data(), *A.basis(1));
    gold[0] = 4.;
    gold[1] = -2.;
    gold[2] = 2.;
    gold[3] = 0.;
    dotk::gtest::checkResults(nrows, gold.data(), *A.basis(2));

    // check Q values
    gold[0] = 0.5;
    gold[1] = 0.5;
    gold[2] = 0.5;
    gold[3] = 0.5;
    dotk::gtest::checkResults(nrows, gold.data(), *Q.basis(0));
    gold[0] = -0.5;
    gold[1] = 0.5;
    gold[2] = 0.5;
    gold[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold.data(), *Q.basis(1));
    gold[0] = 0.5;
    gold[1] = -0.5;
    gold[2] = 0.5;
    gold[3] = -0.5;
    dotk::gtest::checkResults(nrows, gold.data(), *Q.basis(2));

    // check R values
    Real gold_R[] = {2., 3., 2., 5., -2., 4.};
    std::vector<Real> data(R.size());
    R.gather(data.size(), data.data());
    dotk::gtest::checkResults(6, gold_R, data.size(), data.data());
}

TEST(OrthoMethod, arnoldiModifiedGramSchmidtE1)
{
    size_t nrows = 4;
    size_t ncols = 4;
    dotk::StdArray<Real> x(nrows);
    dotk::StdArray<Real> y(nrows + 1);
    dotk::serial::DOTk_ColumnMatrix<Real> A(x, ncols);
    dotk::serial::DOTk_ColumnMatrix<Real> Q(x, ncols + 1);
    dotk::serial::DOTk_ColumnMatrix<Real> Hessenberg(y, ncols);
    // basis 1
    A(0, 0) = 10.;
    A(1, 0) = -1.;
    A(2, 0) = 2.;
    A(3, 0) = 0.;
    // basis 2
    A(0, 1) = -1.;
    A(1, 1) = 11.;
    A(2, 1) = -1.;
    A(3, 1) = 3.;
    // basis 3
    A(0, 2) = 2.;
    A(1, 2) = -1.;
    A(2, 2) = 10.;
    A(3, 2) = -1.;
    // basis 4
    A(0, 3) = 0.;
    A(1, 3) = 3.;
    A(2, 3) = -1.;
    A(3, 3) = 8.;

    dotk::qr::arnoldiModifiedGramSchmidt(A, Q, Hessenberg);

    Real tolerance = 1e-8;
    std::vector<Real> gold(ncols, 0.);
    checkOrthogonality(ncols, Q);

    // basis 1
    gold[0] = 0.5;
    gold[1] = 0.5;
    gold[2] = 0.5;
    gold[3] = 0.5;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(0));
    EXPECT_NEAR(1., Q.basis(0)->norm(), tolerance);

    // basis 2
    gold[0] = 0.15075567228888181;
    gold[1] = 0.75377836144440913;
    gold[2] = -0.45226701686664544;
    gold[3] = -0.45226701686664544;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(1));
    EXPECT_NEAR(1., Q.basis(1)->norm(), tolerance);

    // basis 3
    gold[0] = -0.53437340489744101;
    gold[1] = 0.26718670244872011;
    gold[2] = -0.41747922257612557;
    gold[3] = 0.68466592502484636;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(2));
    EXPECT_NEAR(1., Q.basis(2)->norm(), tolerance);

    // basis 4
    gold[0] = -0.66461853074605248;
    gold[1] = 0.33230926537303329;
    gold[2] = 0.60923365318387857;
    gold[3] = -0.2769243878108586;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(3));
    EXPECT_NEAR(1., Q.basis(3)->norm(), tolerance);

    // basis 5
    gold[0] = -0.14664059954361947;
    gold[1] = -0.78652685209759543;
    gold[2] = 0.47991468941548193;
    gold[3] = 0.35993601706161144;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(4));
    EXPECT_NEAR(1., Q.basis(4)->norm(), tolerance);

    gold.resize(Q.basisDimension(), 0.);
    // Hessenberg 1
    gold[0] = 10.75;
    gold[1] = 0.82915619758884995;
    gold[2] = 0.;
    gold[3] = 0.;
    gold[4] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Hessenberg.basis(0));
    EXPECT_NEAR(1., Q.basis(0)->norm(), tolerance);

    // Hessenberg 2
    gold[0] = 0.82915619758884995;
    gold[1] = 7.8863636363636376;
    gold[2] = 3.2828127427759615;
    gold[3] = 0.;
    gold[4] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Hessenberg.basis(1));
    EXPECT_NEAR(1., Q.basis(1)->norm(), tolerance);

    // Hessenberg 3
    gold[0] = -1.7763568394002505e-15;
    gold[1] = 3.2828127427759584;
    gold[2] = 12.204127161182374;
    gold[3] = 0.63076913190808248;
    gold[4] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Hessenberg.basis(2));
    EXPECT_NEAR(1., Q.basis(2)->norm(), tolerance);

    // Hessenberg 4
    gold[0] = 1.1102230246251565e-14;
    gold[1] = 6.7057470687359455e-14;
    gold[2] = 0.63076913190811701;
    gold[3] = 8.1595092024539859;
    gold[4] = 6.6625222803969918e-14;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Hessenberg.basis(3));
    EXPECT_NEAR(1., Q.basis(3)->norm(), tolerance);
}

TEST(OrthoMethod, Householder)
{
    size_t nrows = 4;
    size_t ncols = 4;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> A(x, ncols);
    dotk::serial::DOTk_ColumnMatrix<Real> Q(x, ncols);
    dotk::serial::DOTk_ColumnMatrix<Real> R(x, ncols);
    // basis 1
    A(0, 0) = 10.;
    A(1, 0) = -1.;
    A(2, 0) = 2.;
    A(3, 0) = 0.;
    // basis 2
    A(0, 1) = -1.;
    A(1, 1) = 11.;
    A(2, 1) = -1.;
    A(3, 1) = 3.;
    // basis 3
    A(0, 2) = 2.;
    A(1, 2) = -1.;
    A(2, 2) = 10.;
    A(3, 2) = -1.;
    // basis 4
    A(0, 3) = 0.;
    A(1, 3) = 3.;
    A(2, 3) = -1.;
    A(3, 3) = 8.;

    dotk::qr::householder(A, Q, R);

    Real tolerance = 1e-8;
    std::vector<Real> gold(ncols, 0.);
    checkOrthogonality(ncols, Q);

    // Q basis 1
    gold[0] = -0.975900072948533;
    gold[1] = 0.097590007294853;
    gold[2] = -0.195180014589707;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(0));
    EXPECT_NEAR(1., Q.basis(0)->norm(), tolerance);

    // Q basis 2
    gold[0] = -0.105653526929161;
    gold[1] = -0.956798339870484;
    gold[2] = 0.049868464710564;
    gold[3] = -0.266246887861486;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(1));
    EXPECT_NEAR(1., Q.basis(1)->norm(), tolerance);

    // Q basis 3
    gold[0] = 0.186345111169031;
    gold[1] = -0.089227790175070;
    gold[2] = -0.976339450932690;
    gold[3] = 0.063837117387371;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(2));
    EXPECT_NEAR(1., Q.basis(2)->norm(), tolerance);

    // Q basis 4
    gold[0] = 0.041615855270295;
    gold[1] = 0.258943099459612;
    gold[2] = -0.078607726621668;
    gold[3] = -0.961788655135703;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q.basis(3));
    EXPECT_NEAR(1., Q.basis(3)->norm(), tolerance);

    // R column 1
    gold[0] = -10.2469507659596;
    gold[1] = 0.;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R.basis(0));

    // R column 2
    gold[0] = 2.244570167781625;
    gold[1] = -11.267737339941178;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R.basis(1));

    // R column 3
    gold[0] = -4.001190299088986;
    gold[1] = 1.510422820979288;
    gold[2] = -9.365313614201140;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R.basis(2));

    // R column 4
    gold[0] = 0.487950036474267;
    gold[1] = -5.050238587213904;
    gold[2] = 1.219353019506445;
    gold[3] = -6.838872216085120;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R.basis(3));
}

}
