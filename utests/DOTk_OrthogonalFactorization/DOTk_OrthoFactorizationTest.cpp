/*
 * DOTk_OrthoFactorizationTest.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_QR.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_Householder.hpp"
#include "DOTk_ColumnMatrix.hpp"
#include "DOTk_ColumnMatrix.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_ModifiedGramSchmidt.hpp"
#include "DOTk_ClassicalGramSchmidt.hpp"

namespace DOTkOrthoFactorizationTest
{

TEST(QR, ClassicalGramSchmidt)
{
    size_t nrows = 4;
    size_t ncols = 4;
    dotk::StdArray<Real> x(nrows);
    std::shared_ptr<dotk::matrix<Real> > A = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    std::shared_ptr<dotk::matrix<Real> > Q = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    std::shared_ptr<dotk::matrix<Real> > R = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    // basis 1
    (*A)(0, 0) = 10.;
    (*A)(1, 0) = -1.;
    (*A)(2, 0) = 2.;
    (*A)(3, 0) = 0.;
    // basis 2
    (*A)(0, 1) = -1.;
    (*A)(1, 1) = 11.;
    (*A)(2, 1) = -1.;
    (*A)(3, 1) = 3.;
    // basis 3
    (*A)(0, 2) = 2.;
    (*A)(1, 2) = -1.;
    (*A)(2, 2) = 10.;
    (*A)(3, 2) = -1.;
    // basis 4
    (*A)(0, 3) = 0.;
    (*A)(1, 3) = 3.;
    (*A)(2, 3) = -1.;
    (*A)(3, 3) = 8.;

    dotk::DOTk_ClassicalGramSchmidt qr;
    EXPECT_EQ(dotk::types::CLASSICAL_GRAM_SCHMIDT_QR, qr.type());
    qr.factorization(A, Q, R);

    // Check Q^t * Q = I
    std::shared_ptr<dotk::matrix<Real> > I = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    Q->gemm(true, false, 1., *Q, 0., *I);
    Real eye[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    dotk::StdArray<Real> data(I->size(), 0.);
    I->gather(data.size(), &(data[0]));
    dotk::gtest::checkResults(I->size(), eye, data.size(), &(data[0]));

    // Check Q * R = A
    std::shared_ptr<dotk::matrix<Real> > Ao = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    Q->gemm(false, false, 1., *R, 0., *Ao);
    Real original[] = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
    data.fill(0.);
    Ao->gather(data.size(), &(data[0]));
    dotk::gtest::checkResults(Ao->size(), original, data.size(), &(data[0]));

    // Q basis 1
    std::vector<Real> gold(ncols, 0.);
    gold[0] = 0.975900072948533;
    gold[1] = -0.097590007294853;
    gold[2] = 0.195180014589707;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(0));
    // Q basis 2
    gold[0] = 0.105653526929161;
    gold[1] = 0.956798339870484;
    gold[2] = -0.049868464710564;
    gold[3] = 0.266246887861486;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(1));
    // Q basis 3
    gold[0] = -0.186345111169031;
    gold[1] = 0.089227790175070;
    gold[2] = 0.976339450932690;
    gold[3] = -0.063837117387371;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(2));
    // Q basis 4
    gold[0] = -0.041615855270295;
    gold[1] = -0.258943099459612;
    gold[2] = 0.078607726621668;
    gold[3] = 0.961788655135703;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(3));

    // R column 1
    gold[0] = 10.2469507659596;
    gold[1] = 0.;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(0));
    // R column 2
    gold[0] = -2.244570167781625;
    gold[1] = 11.267737339941178;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(1));
    // R column 3
    gold[0] = 4.001190299088986;
    gold[1] = -1.510422820979288;
    gold[2] = 9.365313614201140;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(2));
    // R column 4
    gold[0] = -0.487950036474267;
    gold[1] = 5.050238587213904;
    gold[2] = -1.219353019506445;
    gold[3] = 6.838872216085120;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(3));
}

TEST(QR, ModifiedGramSchmidt)
{
    size_t nrows = 4;
    size_t ncols = 4;
    dotk::StdArray<Real> x(nrows);
    std::shared_ptr<dotk::matrix<Real> > A = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    std::shared_ptr<dotk::matrix<Real> > Q = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    std::shared_ptr<dotk::matrix<Real> > R = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    // basis 1
    (*A)(0, 0) = 10.;
    (*A)(1, 0) = -1.;
    (*A)(2, 0) = 2.;
    (*A)(3, 0) = 0.;
    // basis 2
    (*A)(0, 1) = -1.;
    (*A)(1, 1) = 11.;
    (*A)(2, 1) = -1.;
    (*A)(3, 1) = 3.;
    // basis 3
    (*A)(0, 2) = 2.;
    (*A)(1, 2) = -1.;
    (*A)(2, 2) = 10.;
    (*A)(3, 2) = -1.;
    // basis 4
    (*A)(0, 3) = 0.;
    (*A)(1, 3) = 3.;
    (*A)(2, 3) = -1.;
    (*A)(3, 3) = 8.;

    dotk::DOTk_ModifiedGramSchmidt qr;
    EXPECT_EQ(dotk::types::MODIFIED_GRAM_SCHMIDT_QR, qr.type());
    qr.factorization(A, Q, R);

    // Check Q^t * Q = I
    std::shared_ptr<dotk::matrix<Real> > I = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    Q->gemm(true, false, 1., *Q, 0., *I);
    Real eye[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    dotk::StdArray<Real> data(I->size(), 0.);
    I->gather(data.size(), &(data[0]));
    dotk::gtest::checkResults(I->size(), eye, data.size(), &(data[0]));

    // Check Q * R = A
    std::shared_ptr<dotk::matrix<Real> > Ao = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    Q->gemm(false, false, 1., *R, 0., *Ao);
    Real original[] = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
    data.fill(0.);
    Ao->gather(data.size(), &(data[0]));
    dotk::gtest::checkResults(Ao->size(), original, data.size(), &(data[0]));

    // Q basis 1
    std::vector<Real> gold(ncols, 0.);
    gold[0] = 0.975900072948533;
    gold[1] = -0.097590007294853;
    gold[2] = 0.195180014589707;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(0));
    // Q basis 2
    gold[0] = 0.105653526929161;
    gold[1] = 0.956798339870484;
    gold[2] = -0.049868464710564;
    gold[3] = 0.266246887861486;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(1));
    // Q basis 3
    gold[0] = -0.186345111169031;
    gold[1] = 0.089227790175070;
    gold[2] = 0.976339450932690;
    gold[3] = -0.063837117387371;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(2));
    // Q basis 4
    gold[0] = -0.041615855270295;
    gold[1] = -0.258943099459612;
    gold[2] = 0.078607726621668;
    gold[3] = 0.961788655135703;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(3));

    // R column 1
    gold[0] = 10.2469507659596;
    gold[1] = 0.;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(0));
    // R column 2
    gold[0] = -2.244570167781625;
    gold[1] = 11.267737339941178;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(1));
    // R column 3
    gold[0] = 4.001190299088986;
    gold[1] = -1.510422820979288;
    gold[2] = 9.365313614201140;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(2));
    // R column 4
    gold[0] = -0.487950036474267;
    gold[1] = 5.050238587213904;
    gold[2] = -1.219353019506445;
    gold[3] = 6.838872216085120;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(3));
}

TEST(QR, Householder)
{
    size_t nrows = 4;
    size_t ncols = 4;
    dotk::StdArray<Real> x(nrows);
    std::shared_ptr<dotk::matrix<Real> > A = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    std::shared_ptr<dotk::matrix<Real> > Q = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    std::shared_ptr<dotk::matrix<Real> > R = std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    // basis 1
    (*A)(0, 0) = 10.;
    (*A)(1, 0) = -1.;
    (*A)(2, 0) = 2.;
    (*A)(3, 0) = 0.;
    // basis 2
    (*A)(0, 1) = -1.;
    (*A)(1, 1) = 11.;
    (*A)(2, 1) = -1.;
    (*A)(3, 1) = 3.;
    // basis 3
    (*A)(0, 2) = 2.;
    (*A)(1, 2) = -1.;
    (*A)(2, 2) = 10.;
    (*A)(3, 2) = -1.;
    // basis 4
    (*A)(0, 3) = 0.;
    (*A)(1, 3) = 3.;
    (*A)(2, 3) = -1.;
    (*A)(3, 3) = 8.;

    dotk::DOTk_Householder qr;
    EXPECT_EQ(dotk::types::HOUSEHOLDER_QR, qr.type());
    qr.factorization(A, Q, R);

    // Check Q^t * Q = I
    std::shared_ptr<dotk::matrix<Real> > I =
            std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    Q->gemm(true, false, 1., *Q, 0., *I);
    Real eye[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    dotk::StdArray<Real> data(I->size(), 0.);
    I->gather(data.size(), &(data[0]));
    dotk::gtest::checkResults(I->size(), eye, data.size(), &(data[0]));

    // Check Q * R = A
    std::shared_ptr<dotk::matrix<Real> > Ao =
            std::make_shared<dotk::serial::DOTk_ColumnMatrix<Real>>(x, ncols);
    Q->gemm(false, false, 1., *R, 0., *Ao);
    Real original[] = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
    data.fill(0.);
    Ao->gather(data.size(), &(data[0]));
    dotk::gtest::checkResults(Ao->size(), original, data.size(), &(data[0]));

    // Q basis 1
    std::vector<Real> gold(ncols, 0.);
    gold[0] = -0.975900072948533;
    gold[1] = 0.097590007294853;
    gold[2] = -0.195180014589707;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(0));
    // Q basis 2
    gold[0] = -0.105653526929161;
    gold[1] = -0.956798339870484;
    gold[2] = 0.049868464710564;
    gold[3] = -0.266246887861486;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(1));
    // Q basis 3
    gold[0] = 0.186345111169031;
    gold[1] = -0.089227790175070;
    gold[2] = -0.976339450932690;
    gold[3] = 0.063837117387371;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(2));
    // Q basis 4
    gold[0] = 0.041615855270295;
    gold[1] = 0.258943099459612;
    gold[2] = -0.078607726621668;
    gold[3] = -0.961788655135703;
    dotk::gtest::checkResults(gold.size(), gold.data(), *Q->basis(3));

    // R column 1
    gold[0] = -10.2469507659596;
    gold[1] = 0.;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(0));
    // R column 2
    gold[0] = 2.244570167781625;
    gold[1] = -11.267737339941178;
    gold[2] = 0.;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(1));
    // R column 3
    gold[0] = -4.001190299088986;
    gold[1] = 1.510422820979288;
    gold[2] = -9.365313614201140;
    gold[3] = 0.;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(2));
    // R column 4
    gold[0] = 0.487950036474267;
    gold[1] = -5.050238587213904;
    gold[2] = 1.219353019506445;
    gold[3] = -6.838872216085120;
    dotk::gtest::checkResults(gold.size(), gold.data(), *R->basis(3));
}

}
