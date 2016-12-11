/*
 * DOTk_EigenTest.cpp
 *
 *  Created on: Jul 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_EigenQR.hpp"
#include "DOTk_EigenUtils.hpp"
#include "DOTk_Householder.hpp"
#include "DOTk_PowerMethod.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_DenseMatrix.hpp"
#include "DOTk_DenseMatrix.cpp"
#include "DOTk_ColumnMatrix.hpp"
#include "DOTk_ColumnMatrix.cpp"
#include "DOTk_RayleighRitz.hpp"
#include "DOTk_RayleighQuotient.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_OrthogonalFactorization.hpp"

namespace DOTkEigenTest
{

TEST(Eigen, PowerMethod)
{
    size_t nrows = 3;
    size_t ncols = 3;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> A(x, ncols);
    // basis 1
    A(0, 0) = 7.;
    A(1, 0) = 4.;
    A(2, 0) = 1.;
    // basis 2
    A(0, 1) = 4.;
    A(1, 1) = 4.;
    A(2, 1) = 4.;
    // basis 3
    A(0, 2) = 1.;
    A(1, 2) = 4.;
    A(2, 2) = 7.;
    // initial guess
    x[0] = 1.;
    x[1] = 2.;
    x[2] = 3.;

    Real lambda = dotk::eigen::powerMethod(A, x);

    Real tolerance = 1e-8;
    EXPECT_NEAR(11.994146341463415, lambda, tolerance);

    std::vector<Real> gold(nrows);
    gold[0] = 0.577068314017231;
    gold[1] = 0.577350223305954;
    gold[2] = 0.577632132594678;
    dotk::gtest::checkResults(gold.size(), gold.data(), x);
}

TEST(Eigen, RayleighQuotientMethod)
{
    size_t nrows = 3;
    size_t ncols = 3;
    dotk::StdArray<Real> x(nrows);
    dotk::serial::DOTk_ColumnMatrix<Real> A(x, ncols);
    // basis 1
    A(0, 0) = 7.;
    A(1, 0) = 4.;
    A(2, 0) = 1.;
    // basis 2
    A(0, 1) = 4.;
    A(1, 1) = 4.;
    A(2, 1) = 4.;
    // basis 3
    A(0, 2) = 1.;
    A(1, 2) = 4.;
    A(2, 2) = 7.;
    // initial guess
    x[0] = 1.;
    x[1] = 2.;
    x[2] = 3.;

    Real lambda = dotk::eigen::rayleighQuotientMethod(A, x);

    Real tolerance = 1e-8;
    EXPECT_NEAR(11.999996185305159, lambda, tolerance);

    std::vector<Real> gold(nrows);
    gold[0] = 0.577068314017231;
    gold[1] = 0.577350223305954;
    gold[2] = 0.577632132594678;
    dotk::gtest::checkResults(gold.size(), gold.data(), x);
}

TEST(Eigen, QR)
{
    size_t ncols = 4;
    std::tr1::shared_ptr<dotk::Vector<Real> > eigenvalues(new dotk::StdArray<Real>(ncols));
    std::tr1::shared_ptr<dotk::matrix<Real> > matrix(new dotk::serial::DOTk_ColumnMatrix<Real>(*eigenvalues, ncols));
    std::tr1::shared_ptr<dotk::matrix<Real> > eigenvectors(new dotk::serial::DOTk_ColumnMatrix<Real>(*eigenvalues, ncols));
    // basis 1
    (*matrix)(0, 0) = 10.;
    (*matrix)(1, 0) = -1.;
    (*matrix)(2, 0) = 2.;
    (*matrix)(3, 0) = 0.;
    // basis 2
    (*matrix)(0, 1) = -1.;
    (*matrix)(1, 1) = 11.;
    (*matrix)(2, 1) = -1.;
    (*matrix)(3, 1) = 3.;
    // basis 3
    (*matrix)(0, 2) = 2.;
    (*matrix)(1, 2) = -1.;
    (*matrix)(2, 2) = 10.;
    (*matrix)(3, 2) = -1.;
    // basis 4
    (*matrix)(0, 3) = 0.;
    (*matrix)(1, 3) = 3.;
    (*matrix)(2, 3) = -1.;
    (*matrix)(3, 3) = 8.;

    std::tr1::shared_ptr<dotk::DOTk_OrthogonalFactorization> qr(new dotk::DOTk_Householder);
    dotk::DOTk_EigenQR eigen(qr);
    EXPECT_EQ(dotk::types::QR_EIGEN_METHOD, eigen.type());
    EXPECT_EQ(25u, eigen.getMaxNumItr());

    eigen.solve(matrix, eigenvalues, eigenvectors);

    // First Eigenpair
    Real tolerance = 1e-5;
    std::vector<Real> gold(ncols, 0.);
    gold[0] = -0.394852799579398;
    gold[1] = 0.679120369336665;
    gold[2] = -0.462405937267577;
    gold[3] = 0.411178233611698;
    dotk::gtest::checkResults(gold.size(), gold.data(), *eigenvectors->basis(0));
    EXPECT_NEAR(14.073453952359682, (*eigenvalues)[0], tolerance);
    // TODO: EIGENVALUES ARE COMPARED TO MATLAB QR - I WILL IMPROVE THIS BY ADDING SHIFT VERSION

    // Second Eigenpair
    gold[0] = -0.622785527602607;
    gold[1] = -0.492692530443975;
    gold[2] = -0.499584653398748;
    gold[3] = -0.346132100709986;
    dotk::gtest::checkResults(gold.size(), gold.data(), *eigenvectors->basis(1));
    EXPECT_NEAR(10.819078338138118, (*eigenvalues)[1], tolerance);

    // Third Eigenpair
    gold[0] = 0.640162726412377;
    gold[1] = -0.226882245366629;
    gold[2] = -0.708102058064892;
    gold[3] = 0.193151768860483;
    dotk::gtest::checkResults(gold.size(), gold.data(), *eigenvectors->basis(2));
    EXPECT_NEAR(8.143435878088210, (*eigenvalues)[2], tolerance);

    // Fourth Eigenpair
    gold[0] = 0.215455649694608;
    gold[1] = 0.494544276212090;
    gold[2] = -0.187583044629336;
    gold[3] = -0.820863827469650;
    dotk::gtest::checkResults(gold.size(), gold.data(), *eigenvectors->basis(3));
    EXPECT_NEAR(5.964026100329862, (*eigenvalues)[3], tolerance);

    // Check Q^t * Q = I
    std::tr1::shared_ptr<dotk::matrix<Real> > I(new dotk::serial::DOTk_ColumnMatrix<Real>(*eigenvalues, ncols));
    eigenvectors->gemm(true, false, 1., *eigenvectors, 0., *I);
    Real eye[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<Real> data(I->size(), 0.);
    I->gather(data.size(), data.data());
    dotk::gtest::checkResults(I->size(), eye, data.size(), data.data());
}

TEST(Eigen, Power)
{
    size_t nrows = 3;
    std::tr1::shared_ptr<dotk::Vector<Real> > eigenvector(new dotk::StdArray<Real>(nrows));
    std::tr1::shared_ptr<dotk::matrix<Real> > A(new dotk::serial::DOTk_DenseMatrix<Real>(nrows));
    // basis 1
    (*A)(0, 0) = 7.;
    (*A)(1, 0) = 4.;
    (*A)(2, 0) = 1.;
    // basis 2
    (*A)(0, 1) = 4.;
    (*A)(1, 1) = 4.;
    (*A)(2, 1) = 4.;
    // basis 3
    (*A)(0, 2) = 1.;
    (*A)(1, 2) = 4.;
    (*A)(2, 2) = 7.;
    // initial guess
    (*eigenvector)[0] = 1.;
    (*eigenvector)[1] = 2.;
    (*eigenvector)[2] = 3.;

    dotk::DOTk_PowerMethod eigen;
    EXPECT_EQ(dotk::types::POWER_METHOD, eigen.type());
    EXPECT_EQ(10u, eigen.getMaxNumItr());
    Real eigenvalue = 0.;
    eigen.solve(A, eigenvalue, eigenvector);

    std::vector<Real> gold(nrows);
    gold[0] = 0.577068314017231;
    gold[1] = 0.577350223305954;
    gold[2] = 0.577632132594678;
    dotk::gtest::checkResults(gold.size(), gold.data(), *eigenvector);
    Real tolerance = 1e-8;
    EXPECT_NEAR(11.994146341463415, eigenvalue, tolerance);
    EXPECT_NEAR(1e-6, eigen.getRelativeDifferenceTolerance(), tolerance);
}

TEST(Eigen, RayleighQuotient)
{
    size_t nrows = 3;
    std::tr1::shared_ptr<dotk::Vector<Real> > eigenvector(new dotk::StdArray<Real>(nrows));
    std::tr1::shared_ptr<dotk::matrix<Real> > A(new dotk::serial::DOTk_DenseMatrix<Real>(nrows));
    // basis 1
    (*A)(0, 0) = 7.;
    (*A)(1, 0) = 4.;
    (*A)(2, 0) = 1.;
    // basis 2
    (*A)(0, 1) = 4.;
    (*A)(1, 1) = 4.;
    (*A)(2, 1) = 4.;
    // basis 3
    (*A)(0, 2) = 1.;
    (*A)(1, 2) = 4.;
    (*A)(2, 2) = 7.;
    // initial guess
    (*eigenvector)[0] = 1.;
    (*eigenvector)[1] = 2.;
    (*eigenvector)[2] = 3.;

    dotk::DOTk_RayleighQuotient eigen;
    EXPECT_EQ(dotk::types::RAYLEIGH_QUOTIENT_METHOD, eigen.type());
    EXPECT_EQ(10u, eigen.getMaxNumItr());
    Real eigenvalue = 0.;
    eigen.solve(A, eigenvalue, eigenvector);

    std::vector<Real> gold(nrows);
    gold[0] = 0.577068314017231;
    gold[1] = 0.577350223305954;
    gold[2] = 0.577632132594678;
    dotk::gtest::checkResults(gold.size(), gold.data(), *eigenvector);
    Real tolerance = 1e-8;
    EXPECT_NEAR(11.999996185305159, eigenvalue, tolerance);
    EXPECT_NEAR(1e-6, eigen.getRelativeDifferenceTolerance(), tolerance);
}

TEST(Eigen, RayleighRitz)
{
    size_t ncols = 4;
    std::tr1::shared_ptr<dotk::Vector<Real> > eigenvalues(new dotk::StdArray<Real>(ncols));
    std::tr1::shared_ptr<dotk::matrix<Real> > matrix(new dotk::serial::DOTk_ColumnMatrix<Real>(*eigenvalues, ncols));
    std::tr1::shared_ptr<dotk::matrix<Real> > eigenvectors(new dotk::serial::DOTk_ColumnMatrix<Real>(*eigenvalues, ncols));
    // basis 1
    (*matrix)(0, 0) = 10.;
    (*matrix)(1, 0) = -1.;
    (*matrix)(2, 0) = 2.;
    (*matrix)(3, 0) = 0.;
    // basis 2
    (*matrix)(0, 1) = -1.;
    (*matrix)(1, 1) = 11.;
    (*matrix)(2, 1) = -1.;
    (*matrix)(3, 1) = 3.;
    // basis 3
    (*matrix)(0, 2) = 2.;
    (*matrix)(1, 2) = -1.;
    (*matrix)(2, 2) = 10.;
    (*matrix)(3, 2) = -1.;
    // basis 4
    (*matrix)(0, 3) = 0.;
    (*matrix)(1, 3) = 3.;
    (*matrix)(2, 3) = -1.;
    (*matrix)(3, 3) = 8.;

    std::tr1::shared_ptr<dotk::DOTk_OrthogonalFactorization> qr(new dotk::DOTk_Householder);
    std::tr1::shared_ptr<dotk::DOTk_EigenQR> eigen(new dotk::DOTk_EigenQR(qr));
    dotk::DOTk_RayleighRitz method(qr, eigen);

    EXPECT_EQ(dotk::types::RAYLEIGH_RITZ_METHOD, method.type());

    method.solve(matrix, eigenvalues, eigenvectors);
}

}
