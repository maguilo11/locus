/*
 * TRROM_SimpleTest.cpp
 *
 *  Created on: Jul 22, 2016
 */

#include "gtest/gtest.h"
#include "TRROM_UtestUtils.hpp"

#include "TRROM_Data.hpp"
#include "TRROM_Basis.hpp"
#include "TRROM_MOCK_SVD.hpp"
#include "TRROM_LowRankSVD.hpp"
#include "TRROM_Rosenbrock.hpp"
#include "TRROM_EpetraVector.hpp"
#include "TRROM_SerialVector.hpp"
#include "TRROM_ReducedBasis.hpp"
#include "TRROM_MOCK_MatlabQR.hpp"
#include "TRROM_ReducedHessian.hpp"
#include "TRROM_ReducedBasisPDE.hpp"
#include "TRROM_MOCK_IndexMatrix.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_AssemblyMngTypeLP.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_ModifiedGramSchmidt.hpp"
#include "TRROM_SteihaugTointDataMng.hpp"
#include "TRROM_SpectralDecompositionMng.hpp"

namespace SimpleTest
{

TEST(Basis, scale)
{
    int numRows = 10;
    int getNumCols = 3;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix(x, getNumCols);

    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26, 27, 28, 29, 30};
    trrom::SerialVector<double> y(data);
    matrix.copy(y);

    double alpha = 2.;
    matrix.scale(alpha);

    trrom::Basis<double> gold(x, getNumCols);
    y.scale(alpha);
    gold.copy(y);
    trrom::test::checkResults(gold, matrix);
}

TEST(Basis, fill)
{
    int numRows = 10;
    int getNumCols = 3;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix(x, getNumCols);

    matrix.fill(1);

    trrom::Basis<double> gold(x, getNumCols);
    std::vector<double> data =
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    trrom::SerialVector<double> y(data);
    gold.copy(y);
    trrom::test::checkResults(gold, matrix);
}

TEST(Basis, insert)
{
    int numRows = 4;
    int getNumCols = 1;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix(x, getNumCols);
    EXPECT_EQ(4, matrix.getNumRows());

    // number of column vectors less than total number of columns
    x.fill(1.);
    matrix.insert(x);
    EXPECT_EQ(1, matrix.getNumCols());
    EXPECT_EQ(1, matrix.snapshots());
    trrom::test::checkResults(x, matrix.vector(0));

    // number of column vectors == total number of columns
    x.fill(2.);
    matrix.insert(x);
    EXPECT_EQ(2, matrix.getNumCols());
    EXPECT_EQ(2, matrix.snapshots());
    trrom::test::checkResults(x, matrix.vector(1));
}

TEST(Basis, copy)
{
    int numRows = 4;
    int getNumCols = 8;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix1(x, getNumCols);
    trrom::Basis<double> matrix2(x, getNumCols);

    matrix1.fill(1);
    matrix2.update(1., matrix1, 0.);
    trrom::test::checkResults(matrix1, matrix2);
}

TEST(Basis, add)
{
    int numRows = 4;
    int getNumCols = 8;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix1(x, getNumCols);
    trrom::Basis<double> matrix2(x, getNumCols);

    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26, 27, 28, 29, 30, 31, 32};
    trrom::SerialVector<double> y(data);
    matrix1.copy(y);
    matrix2.copy(y);
    matrix1.update(2., matrix2, 1.);

    std::vector<double> gdata = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
                                 69, 72, 75, 78, 81, 84, 87, 90, 93, 96};
    trrom::SerialVector<double> yy(gdata);
    trrom::Basis<double> gold(x, getNumCols);
    gold.copy(yy);
    trrom::test::checkResults(gold, matrix1);
}

TEST(Basis, gemv)
{
    int numRows = 10;
    int getNumCols = 3;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix(x, getNumCols);

    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26, 27, 28, 29, 30};
    trrom::SerialVector<double> y(data);
    matrix.copy(y);

    double beta = 2;
    double alpha = 2;
    trrom::SerialVector<double> input(getNumCols, 1.);
    trrom::SerialVector<double> output(numRows);
    for(int index = 0; index < numRows; ++index)
    {
        output[index] = index + 1;
    }

    matrix.gemv(false, alpha, input, beta, output);

    std::vector<double> gdata = {68, 76, 84, 92, 100, 108, 116, 124, 132, 140};
    trrom::SerialVector<double> gold(gdata);
    trrom::test::checkResults(gold, output);
}

TEST(Basis, gemvT)
{
    int numRows = 10;
    int getNumCols = 3;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> matrix(x, getNumCols);

    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26, 27, 28, 29, 30};
    trrom::SerialVector<double> y(data);
    matrix.copy(y);

    double beta = 2;
    double alpha = 2;
    trrom::SerialVector<double> input(numRows, 1.);
    trrom::SerialVector<double> output(getNumCols);
    output[0] = 1;
    output[1] = 2;
    output[2] = 3;

    matrix.gemv(true, alpha, input, beta, output);

    std::vector<double> gdata = {112, 314, 516};
    trrom::SerialVector<double> gold(gdata);
    trrom::test::checkResults(gold, output);
}

TEST(Basis, gemm1)
{
    int numRows = 2;
    int getNumCols = 2;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> A(x, getNumCols);
    A.replaceGlobalValue(0, 0, 1);
    A.replaceGlobalValue(0, 1, 2);
    A.replaceGlobalValue(1, 0, 3);
    A.replaceGlobalValue(1, 1, 4);
    trrom::Basis<double> B(x, getNumCols);
    B.replaceGlobalValue(0, 0, 5);
    B.replaceGlobalValue(0, 1, 6);
    B.replaceGlobalValue(1, 0, 7);
    B.replaceGlobalValue(1, 1, 8);
    trrom::Basis<double> C(x, getNumCols);

    trrom::Basis<double> gold(x, getNumCols);
    trrom::SerialVector<double> data(numRows * getNumCols, 0.);
    // C = A*B
    A.gemm(false, false, 1., B, 0., C);
    data[0] = 19;
    data[1] = 43;
    data[2] = 22;
    data[3] = 50;
    gold.copy(data);
    trrom::test::checkResults(gold, C);

    // C = A^t*B
    C.fill(0.);
    A.gemm(true, false, 1., B, 0., C);
    data[0] = 26;
    data[1] = 38;
    data[2] = 30;
    data[3] = 44;
    gold.copy(data);
    trrom::test::checkResults(gold, C);

    // C = A*B^t
    C.fill(0.);
    A.gemm(false, true, 1., B, 0., C);
    data[0] = 17;
    data[1] = 39;
    data[2] = 23;
    data[3] = 53;
    gold.copy(data);
    trrom::test::checkResults(gold, C);

    // C = A^t*B^t
    C.fill(0.);
    A.gemm(true, true, 1., B, 0., C);
    data[0] = 23;
    data[1] = 34;
    data[2] = 31;
    data[3] = 46;
    gold.copy(data);
    trrom::test::checkResults(gold, C);
}

TEST(Basis, gemm2)
{
    int numRows = 3;
    int getNumCols = 2;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> A(x, getNumCols);
    A.replaceGlobalValue(0, 0, 1.);
    A.replaceGlobalValue(0, 1, 2.);
    A.replaceGlobalValue(1, 0, 3.);
    A.replaceGlobalValue(1, 1, 4.);
    A.replaceGlobalValue(2, 0, 5.);
    A.replaceGlobalValue(2, 1, 6.);

    numRows = 2;
    getNumCols = 3;
    trrom::SerialVector<double> y(numRows);
    trrom::Basis<double> B(y, getNumCols);
    B.replaceGlobalValue(0, 0, 1.);
    B.replaceGlobalValue(0, 1, 2.);
    B.replaceGlobalValue(0, 2, 3.);
    B.replaceGlobalValue(1, 0, 4.);
    B.replaceGlobalValue(1, 1, 5.);
    B.replaceGlobalValue(1, 2, 6.);

    // C = A*B
    std::tr1::shared_ptr<trrom::Matrix<double> > C = B.create(3, 3);
    std::tr1::shared_ptr<trrom::Matrix<double> > gold = B.create(3, 3);
    trrom::SerialVector<double> data(C->getNumRows() * C->getNumCols(), 0.);
    A.gemm(false, false, 1., B, 0., *C);
    data[0] = 9;
    data[1] = 19;
    data[2] = 29;
    data[3] = 12;
    data[4] = 26;
    data[5] = 40;
    data[6] = 15;
    data[7] = 33;
    data[8] = 51;
    dynamic_cast<trrom::Basis<double>&>(*gold).copy(data);
    trrom::test::checkResults(*gold, *C);

    // C = A*B^t
    A.gemm(false, true, 1., A, 0., *C);
    data[0] = 5;
    data[1] = 11;
    data[2] = 17;
    data[3] = 11;
    data[4] = 25;
    data[5] = 39;
    data[6] = 17;
    data[7] = 39;
    data[8] = 61;
    dynamic_cast<trrom::Basis<double>&>(*gold).copy(data);
    trrom::test::checkResults(*gold, *C);

    // C = A^t*B
    C = B.create(2, 2);
    gold = B.create(2, 2);
    A.gemm(true, false, 1., A, 0., *C);
    std::tr1::shared_ptr<trrom::Vector<double> > z = data.create(4);
    (*z)[0] = 35;
    (*z)[1] = 44;
    (*z)[2] = 44;
    (*z)[3] = 56;
    dynamic_cast<trrom::Basis<double>&>(*gold).copy(*z);
    trrom::test::checkResults(*gold, *C);

    // C = A^t*B^t
    A.gemm(true, true, 1., B, 0., *C);
    (*z)[0] = 22;
    (*z)[1] = 28;
    (*z)[2] = 49;
    (*z)[3] = 64;
    dynamic_cast<trrom::Basis<double>&>(*gold).copy(*z);
    trrom::test::checkResults(*gold, *C);
}

TEST(Basis, gemm3)
{
    int numRows = 3;
    int getNumCols = 2;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> A(x, getNumCols);
    A.replaceGlobalValue(0, 0, 1.);
    A.replaceGlobalValue(0, 1, 2.);
    A.replaceGlobalValue(1, 0, 3.);
    A.replaceGlobalValue(1, 1, 4.);
    A.replaceGlobalValue(2, 0, 5.);
    A.replaceGlobalValue(2, 1, 6.);
    numRows = 2;
    getNumCols = 3;
    trrom::SerialVector<double> y(numRows);
    trrom::Basis<double> B(y, getNumCols);
    B.replaceGlobalValue(0, 0, 1.);
    B.replaceGlobalValue(0, 1, 2.);
    B.replaceGlobalValue(0, 2, 3.);
    B.replaceGlobalValue(1, 0, 4.);
    B.replaceGlobalValue(1, 1, 5.);
    B.replaceGlobalValue(1, 2, 6.);

    // C = A*B
    numRows = 3;
    getNumCols = 3;
    trrom::SerialVector<double> z(numRows);
    trrom::Basis<double> C(z, getNumCols);
    C.fill(1.);
    A.gemm(false, false, 2., B, 1., C);
    trrom::SerialVector<double> data1(numRows * getNumCols, 0.);
    data1[0] = 19;
    data1[1] = 39;
    data1[2] = 59;
    data1[3] = 25;
    data1[4] = 53;
    data1[5] = 81;
    data1[6] = 31;
    data1[7] = 67;
    data1[8] = 103;
    trrom::Basis<double> gold1(z, getNumCols);
    gold1.copy(data1);
    trrom::test::checkResults(gold1, C);

    // C = A*B'
    C.fill(1.);
    A.gemm(false, true, 2., A, 1., C);
    data1[0] = 11;
    data1[1] = 23;
    data1[2] = 35;
    data1[3] = 23;
    data1[4] = 51;
    data1[5] = 79;
    data1[6] = 35;
    data1[7] = 79;
    data1[8] = 123;
    gold1.copy(data1);
    trrom::test::checkResults(gold1, C);

    // C = A'*B
    numRows = 2;
    getNumCols = 2;
    trrom::SerialVector<double> w(numRows);
    trrom::Basis<double> D(w, getNumCols);
    D.fill(1.);
    A.gemm(true, false, 2., A, 1., D);
    trrom::SerialVector<double> data2(numRows * getNumCols, 0.);
    data2[0] = 71;
    data2[1] = 89;
    data2[2] = 89;
    data2[3] = 113;
    trrom::Basis<double> gold2(w, getNumCols);
    gold2.copy(data2);
    trrom::test::checkResults(gold2, D);

    // C = A'*B'
    D.fill(1.);
    A.gemm(true, true, 2., B, 1., D);
    data2[0] = 45;
    data2[1] = 57;
    data2[2] = 99;
    data2[3] = 129;
    gold2.copy(data2);
    trrom::test::checkResults(gold2, D);
}

TEST(ModifiedGramSchmidt, factorize)
{
    int numRows = 4;
    int getNumCols = 4;
    trrom::SerialVector<double> x(numRows);
    trrom::Basis<double> A(x, getNumCols);
    trrom::Basis<double> Q(x, getNumCols);
    trrom::Basis<double> R(x, getNumCols);

    // basis 1
    A.replaceGlobalValue(0, 0, 10.);
    A.replaceGlobalValue(1, 0, -1.);
    A.replaceGlobalValue(2, 0, 2.);
    A.replaceGlobalValue(3, 0, 0.);
    // basis 2
    A.replaceGlobalValue(0, 1, -1.);
    A.replaceGlobalValue(1, 1, 11.);
    A.replaceGlobalValue(2, 1, -1.);
    A.replaceGlobalValue(3, 1, 3.);
    // basis 3
    A.replaceGlobalValue(0, 2, 2.);
    A.replaceGlobalValue(1, 2, -1.);
    A.replaceGlobalValue(2, 2, 10.);
    A.replaceGlobalValue(3, 2, -1.);
    // basis 4
    A.replaceGlobalValue(0, 3, 0.);
    A.replaceGlobalValue(1, 3, 3.);
    A.replaceGlobalValue(2, 3, -1.);
    A.replaceGlobalValue(3, 3, 8.);

    trrom::ModifiedGramSchmidt method;
    EXPECT_EQ(trrom::types::MODIFIED_GRAM_SCHMIDT_QR, method.type());
    method.factorize(A, Q, R);

    // Check Q^t * Q = I
    trrom::Basis<double> I(x, getNumCols);
    Q.gemm(true, false, 1., Q, 0., I);
    trrom::Basis<double> mgold(x, getNumCols);
    mgold.replaceGlobalValue(0, 0, 1.);
    mgold.replaceGlobalValue(1, 1, 1.);
    mgold.replaceGlobalValue(2, 2, 1.);
    mgold.replaceGlobalValue(3, 3, 1.);
    trrom::test::checkResults(mgold, I);

    // Check Q * R = A
    trrom::Basis<double> Ao(x, getNumCols);
    Q.gemm(false, false, 1., R, 0., Ao);
    mgold.fill(0);
    mgold.replaceGlobalValue(0, 0, 10);
    mgold.replaceGlobalValue(1, 0, -1);
    mgold.replaceGlobalValue(2, 0, 2);
    mgold.replaceGlobalValue(3, 0, 0);
    mgold.replaceGlobalValue(0, 1, -1);
    mgold.replaceGlobalValue(1, 1, 11);
    mgold.replaceGlobalValue(2, 1, -1);
    mgold.replaceGlobalValue(3, 1, 3);
    mgold.replaceGlobalValue(0, 2, 2);
    mgold.replaceGlobalValue(1, 2, -1);
    mgold.replaceGlobalValue(2, 2, 10);
    mgold.replaceGlobalValue(3, 2, -1);
    mgold.replaceGlobalValue(0, 3, 0);
    mgold.replaceGlobalValue(1, 3, 3);
    mgold.replaceGlobalValue(2, 3, -1);
    mgold.replaceGlobalValue(3, 3, 8);
    trrom::test::checkResults(mgold, Ao);

    // Q basis 1
    trrom::SerialVector<double> gold(getNumCols);
    gold[0] = 0.975900072948533;
    gold[1] = -0.097590007294853;
    gold[2] = 0.195180014589707;
    gold[3] = 0.;
    trrom::test::checkResults(gold, Q.vector(0));
    // Q basis 2
    gold[0] = 0.105653526929161;
    gold[1] = 0.956798339870484;
    gold[2] = -0.049868464710564;
    gold[3] = 0.266246887861486;
    trrom::test::checkResults(gold, Q.vector(1));
    // Q basis 3
    gold[0] = -0.186345111169031;
    gold[1] = 0.089227790175070;
    gold[2] = 0.976339450932690;
    gold[3] = -0.063837117387371;
    trrom::test::checkResults(gold, Q.vector(2));
    // Q basis 4
    gold[0] = -0.041615855270295;
    gold[1] = -0.258943099459612;
    gold[2] = 0.078607726621668;
    gold[3] = 0.961788655135703;
    trrom::test::checkResults(gold, Q.vector(3));

    // R column 1
    gold[0] = 10.2469507659596;
    gold[1] = 0.;
    gold[2] = 0.;
    gold[3] = 0.;
    trrom::test::checkResults(gold, R.vector(0));
    // R column 2
    gold[0] = -2.244570167781625;
    gold[1] = 11.267737339941178;
    gold[2] = 0.;
    gold[3] = 0.;
    trrom::test::checkResults(gold, R.vector(1));
    // R column 3
    gold[0] = 4.001190299088986;
    gold[1] = -1.510422820979288;
    gold[2] = 9.365313614201140;
    gold[3] = 0.;
    trrom::test::checkResults(gold, R.vector(2));
    // R column 4
    gold[0] = -0.487950036474267;
    gold[1] = 5.050238587213904;
    gold[2] = -1.219353019506445;
    gold[3] = 6.838872216085120;
    trrom::test::checkResults(gold, R.vector(3));
}

TEST(ReducedBasis, pod)
{
    int ndofs = 6;
    int nsnap = 3;
    trrom::SerialVector<double> x(ndofs);
    trrom::Basis<double> basis(x, nsnap);
    trrom::Basis<double> snapshots(x, nsnap);
    trrom::SerialVector<double> y(nsnap);
    trrom::SerialVector<double> singular_val(nsnap);
    trrom::Basis<double> singular_vec(y, nsnap);

    // Set snapshots for unit test
    snapshots.replaceGlobalValue(0, 0, 0.792207329559554);
    snapshots.replaceGlobalValue(0, 1, 0.678735154857774);
    snapshots.replaceGlobalValue(0, 2, 0.706046088019609);
    snapshots.replaceGlobalValue(1, 0, 0.959492426392903);
    snapshots.replaceGlobalValue(1, 1, 0.757740130578333);
    snapshots.replaceGlobalValue(1, 2, 0.031832846377421);
    snapshots.replaceGlobalValue(2, 0, 0.655740699156587);
    snapshots.replaceGlobalValue(2, 1, 0.743132468124916);
    snapshots.replaceGlobalValue(2, 2, 0.276922984960890);
    snapshots.replaceGlobalValue(3, 0, 0.035711678574190);
    snapshots.replaceGlobalValue(3, 1, 0.392227019534168);
    snapshots.replaceGlobalValue(3, 2, 0.046171390631154);
    snapshots.replaceGlobalValue(4, 0, 0.849129305868777);
    snapshots.replaceGlobalValue(4, 1, 0.655477890177557);
    snapshots.replaceGlobalValue(4, 2, 0.097131781235848);
    snapshots.replaceGlobalValue(5, 0, 0.933993247757551);
    snapshots.replaceGlobalValue(5, 1, 0.171186687811562);
    snapshots.replaceGlobalValue(5, 2, 0.823457828327293);
    // Set singular values for unit test
    singular_val[0] = 0.176291651111878;
    singular_val[1] = 0.695058842537319;
    singular_val[2] = 6.167248315069438;
    // Set singular vectors for unit test
    singular_vec.replaceGlobalValue(0, 0, -0.655755263680298);
    singular_vec.replaceGlobalValue(0, 1, 0.057661128661978);
    singular_vec.replaceGlobalValue(0, 2, 0.752768376326350);
    singular_vec.replaceGlobalValue(1, 0, 0.584312694702689);
    singular_vec.replaceGlobalValue(1, 1, -0.592633544619941);
    singular_vec.replaceGlobalValue(1, 2, 0.554404325921513);
    singular_vec.replaceGlobalValue(2, 0, 0.478083370307800);
    singular_vec.replaceGlobalValue(2, 1, 0.803405673388377);
    singular_vec.replaceGlobalValue(2, 2, 0.354930436849960);

    double threshold = 0.9999;
    trrom::properOrthogonalDecomposition(threshold, singular_val, singular_vec, snapshots, basis);

    trrom::SerialVector<double> data(ndofs * nsnap, 0.);
    data[0] = 0.511224489804523;
    data[6] = 0.252703973946664;
    data[12] = 0.492567560436969;
    data[1] = -0.407783056434706;
    data[7] = -0.441599231055225;
    data[13] = 0.464553014644157;
    data[2] = 0.325358493436934;
    data[8] = -0.216040031438741;
    data[14] = 0.404247132941033;
    data[3] = 0.542640199339754;
    data[9] = -0.231849487408929;
    data[15] = 0.104986384154719;
    data[4] = -0.303378590570133;
    data[10] = -0.313613547154724;
    data[16] = 0.417602795764322;
    data[5] = -0.282855785695668;
    data[11] = 0.736444080516773;
    data[17] = 0.439019036729722;
    trrom::Basis<double> gold(x, nsnap);
    gold.copy(data);
    trrom::test::checkResults(gold, basis, 1e-6);
}

TEST(LowRankSVD, formMatrixK)
{
    int rank = 3;
    int nsnap = 4;
    trrom::SerialVector<double> y(rank);
    trrom::Basis<double> M(y, nsnap);
    trrom::SerialVector<double> singular_val(rank);
    trrom::SerialVector<double> x(nsnap);
    trrom::Basis<double> R(x, nsnap);
    trrom::SerialVector<double> z(rank + nsnap);
    trrom::Basis<double> K(z, rank + nsnap);

    // Set singular values for unit test
    singular_val[0] = 0.176291651111878;
    singular_val[1] = 0.695058842537319;
    singular_val[2] = 6.167248315069438;
    // Set upper triangular matrix R
    R.replaceGlobalValue(0, 0, 10.2469507659596);
    R.replaceGlobalValue(0, 1, -2.244570167781625);
    R.replaceGlobalValue(0, 2, 4.001190299088986);
    R.replaceGlobalValue(0, 3, -0.487950036474267);
    R.replaceGlobalValue(1, 0, 0.);
    R.replaceGlobalValue(1, 1, 11.267737339941178);
    R.replaceGlobalValue(1, 2, -1.510422820979288);
    R.replaceGlobalValue(1, 3, 5.050238587213904);
    R.replaceGlobalValue(2, 0, 0.);
    R.replaceGlobalValue(2, 1, 0.);
    R.replaceGlobalValue(2, 2, 9.365313614201140);
    R.replaceGlobalValue(2, 3, -1.219353019506445);
    R.replaceGlobalValue(3, 0, 0.);
    R.replaceGlobalValue(3, 1, 0.);
    R.replaceGlobalValue(3, 2, 0.);
    R.replaceGlobalValue(3, 3, 6.838872216085120);
    // Set matrix M
    M.replaceGlobalValue(0, 0, 10.2469507659596);
    M.replaceGlobalValue(0, 1, -2.244570167781625);
    M.replaceGlobalValue(0, 2, 4.001190299088986);
    M.replaceGlobalValue(0, 3, -0.487950036474267);
    M.replaceGlobalValue(1, 0, 0.);
    M.replaceGlobalValue(1, 1, 11.267737339941178);
    M.replaceGlobalValue(1, 2, -1.510422820979288);
    M.replaceGlobalValue(1, 3, 5.050238587213904);
    M.replaceGlobalValue(2, 0, 0.);
    M.replaceGlobalValue(2, 1, 0.);
    M.replaceGlobalValue(2, 2, 9.365313614201140);
    M.replaceGlobalValue(2, 3, -1.219353019506445);

    trrom::LowRankSVD svd;
    svd.formMatrixK(singular_val, M, R, K);

    int dim = rank + nsnap;
    trrom::SerialVector<double> data(dim * dim, 0.);
    data[0] = 0.1762916511;
    data[7] = 0;
    data[14] = 0;
    data[21] = 10.24695076;
    data[28] = -2.2445701677;
    data[35] = 4.0011902990;
    data[42] = -0.4879500364;
    data[1] = 0;
    data[8] = 0.6950588425;
    data[15] = 0;
    data[22] = 0;
    data[29] = 11.2677373399;
    data[36] = -1.5104228209;
    data[43] = 5.0502385872;
    data[2] = 0;
    data[9] = 0;
    data[16] = 6.1672483150;
    data[23] = 0;
    data[30] = 0;
    data[37] = 9.3653136142;
    data[44] = -1.2193530195;
    data[3] = 0;
    data[10] = 0;
    data[17] = 0;
    data[24] = 10.24695076;
    data[31] = -2.2445701677;
    data[38] = 4.0011902990;
    data[45] = -0.4879500364;
    data[4] = 0;
    data[11] = 0;
    data[18] = 0;
    data[25] = 0;
    data[32] = 11.2677373399;
    data[39] = -1.5104228209;
    data[46] = 5.0502385872;
    data[5] = 0;
    data[12] = 0;
    data[19] = 0;
    data[26] = 0;
    data[33] = 0;
    data[40] = 9.3653136142;
    data[47] = -1.2193530195;
    data[6] = 0;
    data[13] = 0;
    data[20] = 0;
    data[27] = 0;
    data[34] = 0;
    data[41] = 0;
    data[48] = 6.8388722160;
    trrom::Basis<double> gold(z, rank + nsnap);
    gold.copy(data);
    trrom::test::checkResults(gold, K, 1e-6);
}

TEST(LowRankSVD, formMatrixUbar)
{
    int rank = 3;
    int nsnap = 4;
    int ndofs = 4;
    trrom::SerialVector<double> x(ndofs);
    trrom::Basis<double> U(x, rank);
    trrom::Basis<double> Q(x, nsnap);
    trrom::SerialVector<double> y(rank + nsnap);
    trrom::Basis<double> C(y, rank + nsnap);
    trrom::Basis<double> Ubar(x, rank + nsnap);

    // Set singular vectors
    U.replaceGlobalValue(0, 0, 10.2469507659596);
    U.replaceGlobalValue(0, 1, -2.244570167781625);
    U.replaceGlobalValue(0, 2, 4.001190299088986);
    U.replaceGlobalValue(1, 0, 0.);
    U.replaceGlobalValue(1, 1, 11.267737339941178);
    U.replaceGlobalValue(1, 2, -1.510422820979288);
    U.replaceGlobalValue(2, 0, 0.);
    U.replaceGlobalValue(2, 1, 0.);
    U.replaceGlobalValue(2, 2, 9.365313614201140);
    U.replaceGlobalValue(3, 0, 0.);
    U.replaceGlobalValue(3, 1, 0.);
    U.replaceGlobalValue(3, 2, 0.);
    // Set orthonormal matrix Q
    Q.replaceGlobalValue(0, 0, 0.9759000729);
    Q.replaceGlobalValue(0, 1, 0.1056535269);
    Q.replaceGlobalValue(0, 2, -0.1863451111);
    Q.replaceGlobalValue(0, 3, -0.0416158552);
    Q.replaceGlobalValue(1, 0, -0.0975900072);
    Q.replaceGlobalValue(1, 1, 0.9567983398);
    Q.replaceGlobalValue(1, 2, 0.0892277901);
    Q.replaceGlobalValue(1, 3, -0.2589430994);
    Q.replaceGlobalValue(2, 0, 0.1951800145);
    Q.replaceGlobalValue(2, 1, -0.0498684647);
    Q.replaceGlobalValue(2, 2, 0.9763394509);
    Q.replaceGlobalValue(2, 3, 0.0786077266);
    Q.replaceGlobalValue(3, 0, 0.);
    Q.replaceGlobalValue(3, 1, 0.2662468878);
    Q.replaceGlobalValue(3, 2, -0.0638371173);
    Q.replaceGlobalValue(3, 3, 0.9617886551);
    // Set matrix C
    C.replaceGlobalValue(0, 0, 1.);
    C.replaceGlobalValue(1, 1, 1.);
    C.replaceGlobalValue(2, 2, 1.);
    C.replaceGlobalValue(3, 3, 1.);
    C.replaceGlobalValue(4, 4, 1.);
    C.replaceGlobalValue(5, 5, 1.);
    C.replaceGlobalValue(6, 6, 1.);

    trrom::LowRankSVD svd;
    svd.formMatrixUbar(U, Q, C, Ubar);

    int dim = rank + nsnap;
    trrom::SerialVector<double> data(ndofs * dim, 0.);
    data[0] = 10.2469507659;
    data[4] = -2.2445701677;
    data[8] = 4.0011902990;
    data[12] = 0.9759000729;
    data[16] = 0.1056535269;
    data[20] = -0.1863451111;
    data[24] = -0.0416158552;
    data[1] = 0;
    data[5] = 11.2677373399;
    data[9] = -1.5104228209;
    data[13] = -0.0975900072;
    data[17] = 0.9567983398;
    data[21] = 0.0892277901;
    data[25] = -0.2589430994;
    data[2] = 0;
    data[6] = 0;
    data[10] = 9.3653136142;
    data[14] = 0.1951800145;
    data[18] = -0.0498684647;
    data[22] = 0.9763394509;
    data[26] = 0.0786077266;
    data[3] = 0;
    data[7] = 0;
    data[11] = 0;
    data[15] = 0.;
    data[19] = 0.2662468878;
    data[23] = -0.0638371173;
    data[27] = 0.9617886551;
    trrom::Basis<double> gold(x, rank + nsnap);
    gold.copy(data);
    trrom::test::checkResults(gold, Ubar, 1e-6);
}

TEST(LowRankSVD, formMatrixVbar)
{
    int rank = 3;
    int num_new_snap = 3;
    int num_old_snap = 4;
    trrom::SerialVector<double> x(num_old_snap);
    trrom::Basis<double> V(x, rank);
    trrom::SerialVector<double> y(num_new_snap + num_old_snap);
    trrom::Basis<double> Vbar(y, rank + num_new_snap);
    trrom::SerialVector<double> z(rank + num_new_snap);
    trrom::Basis<double> D(z, rank + num_new_snap);

    // Set singular vectors
    V.replaceGlobalValue(0, 0, 10.2469507659596);
    V.replaceGlobalValue(0, 1, -2.244570167781625);
    V.replaceGlobalValue(0, 2, 4.001190299088986);
    V.replaceGlobalValue(1, 0, 0.);
    V.replaceGlobalValue(1, 1, 11.267737339941178);
    V.replaceGlobalValue(1, 2, -1.510422820979288);
    V.replaceGlobalValue(2, 0, 0.);
    V.replaceGlobalValue(2, 1, 0.);
    V.replaceGlobalValue(2, 2, 9.365313614201140);
    V.replaceGlobalValue(3, 0, 0.);
    V.replaceGlobalValue(3, 1, 0.);
    V.replaceGlobalValue(3, 2, 0.);
    // Set matrix C
    D.replaceGlobalValue(0, 0, 2.);
    D.replaceGlobalValue(1, 1, 2.);
    D.replaceGlobalValue(2, 2, 2.);
    D.replaceGlobalValue(3, 3, 2.);
    D.replaceGlobalValue(4, 4, 2.);
    D.replaceGlobalValue(5, 5, 2.);

    trrom::LowRankSVD svd;
    svd.formMatrixVbar(num_new_snap, V, D, Vbar);

    int numRows = rank + num_old_snap;
    int getNumCols = rank + num_new_snap;
    trrom::SerialVector<double> data(numRows * getNumCols, 0.);
    data[0] = 20.4939015319;
    data[7] = -4.4891403355;
    data[14] = 8.0023805981;
    data[21] = 0;
    data[28] = 0;
    data[35] = 0;
    data[1] = 0;
    data[8] = 22.5354746798;
    data[15] = -3.0208456419;
    data[22] = 0;
    data[29] = 0;
    data[36] = 0;
    data[2] = 0;
    data[9] = 0;
    data[16] = 18.730627228;
    data[23] = 0;
    data[30] = 0;
    data[37] = 0;
    data[3] = 0;
    data[10] = 0;
    data[17] = 0;
    data[24] = 0;
    data[31] = 0;
    data[38] = 0;
    data[4] = 0;
    data[11] = 0;
    data[18] = 0;
    data[25] = 2;
    data[32] = 0;
    data[39] = 0;
    data[5] = 0;
    data[12] = 0;
    data[19] = 0;
    data[26] = 0;
    data[33] = 2;
    data[40] = 0;
    data[6] = 0;
    data[13] = 0;
    data[20] = 0;
    data[27] = 0;
    data[34] = 0;
    data[41] = 2;
    trrom::Basis<double> gold(y, rank + num_new_snap);
    gold.copy(data);
    trrom::test::checkResults(gold, Vbar, 1e-6);
}

TEST(LowRankSVD, solve)
{
    std::tr1::shared_ptr<trrom::mock::SVD> svd(new trrom::mock::SVD);
    std::tr1::shared_ptr<trrom::mock::MatlabQR> qr(new trrom::mock::MatlabQR);
    trrom::LowRankSVD thin_svd(svd, qr);

    int NumRows = 4;
    int NumCols = 2;
    trrom::SerialVector<double> x(NumRows);
    std::tr1::shared_ptr<trrom::Basis<double> > snapshots(new trrom::Basis<double>(x, NumCols));
    (*snapshots).replaceGlobalValue(0, 0, 2.);
    (*snapshots).replaceGlobalValue(0, 1, 0.);
    (*snapshots).replaceGlobalValue(1, 0, -1);
    (*snapshots).replaceGlobalValue(1, 1, 3.);
    (*snapshots).replaceGlobalValue(2, 0, 10);
    (*snapshots).replaceGlobalValue(2, 1, -1);
    (*snapshots).replaceGlobalValue(3, 0, -1);
    (*snapshots).replaceGlobalValue(3, 1, 8.);

    const int NUM_SINGULAR_VALUES = 2;
    std::tr1::shared_ptr<trrom::Vector<double> >
        singular_values(new trrom::SerialVector<double>(NUM_SINGULAR_VALUES, 0.));
    (*singular_values)[0] = 11.806131215443042;
    (*singular_values)[1] = 9.466533986826510;

    std::tr1::shared_ptr<trrom::Matrix<double> > left_singular_vectors = snapshots->create(4, 2);
    // Column 1
    (*left_singular_vectors).replaceGlobalValue(0, 0, 0.541332911177864);
    (*left_singular_vectors).replaceGlobalValue(1, 0, -0.821530520345489);
    (*left_singular_vectors).replaceGlobalValue(2, 0, 0.164589314655955);
    (*left_singular_vectors).replaceGlobalValue(3, 0, 0.070403415525477);
    // Column 2
    (*left_singular_vectors).replaceGlobalValue(0, 1, 0.819300293644437);
    (*left_singular_vectors).replaceGlobalValue(1, 1, 0.558245394399770);
    (*left_singular_vectors).replaceGlobalValue(2, 1, 0.116874706526009);
    (*left_singular_vectors).replaceGlobalValue(3, 1, -0.058731690253598);
    std::tr1::shared_ptr<trrom::Matrix<double> > right_singular_vectors = snapshots->create(2, 2);
    // Column 1
    (*right_singular_vectors).replaceGlobalValue(0, 0, 0.555985541889449);
    (*right_singular_vectors).replaceGlobalValue(1, 0, -0.831191961709145);
    // Column 2
    (*right_singular_vectors).replaceGlobalValue(0, 1, 0.831191961709145);
    (*right_singular_vectors).replaceGlobalValue(1, 1, 0.555985541889449);

    thin_svd.solve(snapshots, singular_values, left_singular_vectors, right_singular_vectors);

    std::tr1::shared_ptr<trrom::Vector<double> > gold_singular_values = singular_values->create();
    (*gold_singular_values)[0] = 13.337774133491964;
    (*gold_singular_values)[1] = 10.205479188810592;
    (*gold_singular_values)[2] = 8.616542289422034;
    (*gold_singular_values)[3] = 7.259970706918076;
    trrom::test::checkResults(*gold_singular_values, *singular_values, 1e-6);

    std::tr1::shared_ptr<trrom::Matrix<double> > gold = left_singular_vectors->create();
    // Column 1
    (*gold).replaceGlobalValue(0, 0, 0.515062341395141);
    (*gold).replaceGlobalValue(1, 0, -0.626687172996972);
    (*gold).replaceGlobalValue(2, 0, 0.560102103374287);
    (*gold).replaceGlobalValue(3, 0, -0.168105935270941);
    // Column 2
    (*gold).replaceGlobalValue(0, 1, -0.494929865714162);
    (*gold).replaceGlobalValue(1, 1, -0.758177136554344);
    (*gold).replaceGlobalValue(2, 1, -0.417017162869087);
    (*gold).replaceGlobalValue(3, 1, -0.079426340108521);
    // Column 3
    (*gold).replaceGlobalValue(0, 2, 0.548509721924528);
    (*gold).replaceGlobalValue(1, 2, -0.160444321628943);
    (*gold).replaceGlobalValue(2, 2, -0.485317907230518);
    (*gold).replaceGlobalValue(3, 2, 0.661710838306806);
    // Column 4
    (*gold).replaceGlobalValue(0, 3, 0.434617415038783);
    (*gold).replaceGlobalValue(1, 3, 0.081781638919421);
    (*gold).replaceGlobalValue(2, 3, -0.526164279089438);
    (*gold).replaceGlobalValue(3, 3, -0.726340565775416);
    trrom::test::checkResults(*gold, *left_singular_vectors, 1e-6);

    gold = right_singular_vectors->create();
    // Column 1
    (*gold).replaceGlobalValue(0, 0, 0.517141370416288);
    (*gold).replaceGlobalValue(1, 0, -0.584851515282251);
    (*gold).replaceGlobalValue(2, 0, 0.556759977375391);
    (*gold).replaceGlobalValue(3, 0, -0.283781316631262);
    // Column 2
    (*gold).replaceGlobalValue(0, 1, -0.492397833884674);
    (*gold).replaceGlobalValue(1, 1, -0.720061743054955);
    (*gold).replaceGlobalValue(2, 1, -0.423539924337457);
    (*gold).replaceGlobalValue(3, 1, -0.244273191051664);
    // Column 3
    (*gold).replaceGlobalValue(0, 2, 0.542550082084813);
    (*gold).replaceGlobalValue(1, 2, -0.288954676631222);
    (*gold).replaceGlobalValue(2, 2, -0.494099141178771);
    (*gold).replaceGlobalValue(3, 2, 0.614825700478688);
    // Column 4
    (*gold).replaceGlobalValue(0, 3, 0.442434836579814);
    (*gold).replaceGlobalValue(1, 3, 0.236569199418821);
    (*gold).replaceGlobalValue(2, 3, -0.516234732240650);
    (*gold).replaceGlobalValue(3, 3, -0.694109595449707);
    trrom::test::checkResults(*gold, *right_singular_vectors, 1e-6);
}

TEST(ReducedBasisDataMng, allocate_vector_functions_test1)
{
    int index_base = 0;
    int element_size = 1;
    int global_num_rows = 10000;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_rows, element_size, index_base, comm);
    trrom::EpetraVector x(map);

    trrom::ReducedBasisData mng;

    mng.allocateReducedDualSolution(x);
    std::tr1::shared_ptr<trrom::Vector<double> > a_copy = mng.createReducedDualSolutionCopy();
    EXPECT_EQ(global_num_rows, dynamic_cast< trrom::EpetraVector* >(a_copy.get())->getNumGlobalElements());

    mng.allocateReducedDualRightHandSide(x);
    std::tr1::shared_ptr<trrom::Vector<double> > b_copy = mng.createReducedDualRightHandSideCopy();
    EXPECT_EQ(global_num_rows, dynamic_cast< trrom::EpetraVector* >(b_copy.get())->getNumGlobalElements());

    mng.allocateReducedStateSolution(x);
    std::tr1::shared_ptr<trrom::Vector<double> > c_copy = mng.createReducedStateSolutionCopy();
    EXPECT_EQ(global_num_rows, dynamic_cast< trrom::EpetraVector* >(c_copy.get())->getNumGlobalElements());

    mng.allocateReducedStateRightHandSide(x);
    std::tr1::shared_ptr<trrom::Vector<double> > d_copy = mng.createReducedStateRightHandSideCopy();
    EXPECT_EQ(global_num_rows, dynamic_cast< trrom::EpetraVector* >(d_copy.get())->getNumGlobalElements());

    mng.allocateLeftHandSideSnapshot(x);
    std::tr1::shared_ptr<trrom::Vector<double> > e_copy = mng.createLeftHandSideSnapshotCopy();
    EXPECT_EQ(global_num_rows, dynamic_cast< trrom::EpetraVector* >(e_copy.get())->getNumGlobalElements());

    mng.allocateLeftHandSideDeimCoefficients(x);
    std::tr1::shared_ptr<trrom::Vector<double> > f_copy = mng.createLeftHandSideDeimCoefficientsCopy();
    EXPECT_EQ(global_num_rows, dynamic_cast< trrom::EpetraVector* >(f_copy.get())->getNumGlobalElements());
}

TEST(SpectralDecompositionMng, allocation)
{
    std::tr1::shared_ptr<trrom::mock::SVD> svd(new trrom::mock::SVD);
    std::tr1::shared_ptr<trrom::mock::MatlabQR> qr(new trrom::mock::MatlabQR);
    trrom::SpectralDecompositionMng mng(svd, qr);

    EXPECT_FALSE(mng.areDualSnapshotsCollected());
    EXPECT_FALSE(mng.areStateSnapshotsCollected());
    EXPECT_FALSE(mng.areLeftHandSideSnapshotsCollected());

    double tolerance = 1e-8;
    EXPECT_NEAR(0.99, mng.getDualBasisEnergyThreshold(), tolerance);
    EXPECT_NEAR(0.99, mng.getDualBasisEnergyThreshold(), tolerance);
    EXPECT_NEAR(0.99, mng.getDualBasisEnergyThreshold(), tolerance);

    mng.setDualBasisEnergyThreshold(0.95);
    EXPECT_NEAR(0.95, mng.getDualBasisEnergyThreshold(), tolerance);
    mng.setStateBasisEnergyThreshold(0.94);
    EXPECT_NEAR(0.94, mng.getStateBasisEnergyThreshold(), tolerance);
    mng.setLeftHandSideBasisEnergyThreshold(0.93);
    EXPECT_NEAR(0.93, mng.getLeftHandSideBasisEnergyThreshold(), tolerance);

    int numRows = 6;
    trrom::SerialVector<double> x(numRows);
    int NumCols = 8;
    trrom::Basis<double> dual_snapshots(x, NumCols);
    mng.setDualSnapshotEnsemble(dual_snapshots);
    EXPECT_EQ(NumCols, mng.getNumDualSnapshots());

    NumCols = 4;
    trrom::Basis<double> state_snapshots(x, NumCols);
    mng.setStateSnapshotEnsemble(state_snapshots);
    EXPECT_EQ(NumCols, mng.getNumStateSnapshots());

    NumCols = 6;
    trrom::Basis<double> lhs_snapshots(x, NumCols);
    mng.setLeftHandSideSnapshotEnsemble(lhs_snapshots);
    EXPECT_EQ(NumCols, mng.getNumLeftHandSideSnapshots());
}

TEST(SpectralDecompositionMng, dual_pod)
{
    int ndofs = 6;
    int num_singular_values = 3;
    trrom::SerialVector<double> x(ndofs);
    trrom::Basis<double> basis(x, num_singular_values);
    trrom::Basis<double> snapshots(x, num_singular_values);

    trrom::SerialVector<double> singular_values(num_singular_values);

    trrom::SerialVector<double> y(num_singular_values);
    trrom::Basis<double> left_singular_vectors(y, num_singular_values);

    // Set snapshots for unit test
    snapshots.replaceGlobalValue(0, 0, 0.792207329559554);
    snapshots.replaceGlobalValue(0, 1, 0.678735154857774);
    snapshots.replaceGlobalValue(0, 2, 0.706046088019609);
    snapshots.replaceGlobalValue(1, 0, 0.959492426392903);
    snapshots.replaceGlobalValue(1, 1, 0.757740130578333);
    snapshots.replaceGlobalValue(1, 2, 0.031832846377421);
    snapshots.replaceGlobalValue(2, 0, 0.655740699156587);
    snapshots.replaceGlobalValue(2, 1, 0.743132468124916);
    snapshots.replaceGlobalValue(2, 2, 0.276922984960890);
    snapshots.replaceGlobalValue(3, 0, 0.035711678574190);
    snapshots.replaceGlobalValue(3, 1, 0.392227019534168);
    snapshots.replaceGlobalValue(3, 2, 0.046171390631154);
    snapshots.replaceGlobalValue(4, 0, 0.849129305868777);
    snapshots.replaceGlobalValue(4, 1, 0.655477890177557);
    snapshots.replaceGlobalValue(4, 2, 0.097131781235848);
    snapshots.replaceGlobalValue(5, 0, 0.933993247757551);
    snapshots.replaceGlobalValue(5, 1, 0.171186687811562);
    snapshots.replaceGlobalValue(5, 2, 0.823457828327293);
    // Set singular values for unit test
    singular_values[0] = 0.176291651111878;
    singular_values[1] = 0.695058842537319;
    singular_values[2] = 6.167248315069438;
    // Set singular vectors for unit test
    left_singular_vectors.replaceGlobalValue(0, 0, -0.655755263680298);
    left_singular_vectors.replaceGlobalValue(0, 1, 0.057661128661978);
    left_singular_vectors.replaceGlobalValue(0, 2, 0.752768376326350);
    left_singular_vectors.replaceGlobalValue(1, 0, 0.584312694702689);
    left_singular_vectors.replaceGlobalValue(1, 1, -0.592633544619941);
    left_singular_vectors.replaceGlobalValue(1, 2, 0.554404325921513);
    left_singular_vectors.replaceGlobalValue(2, 0, 0.478083370307800);
    left_singular_vectors.replaceGlobalValue(2, 1, 0.803405673388377);
    left_singular_vectors.replaceGlobalValue(2, 2, 0.354930436849960);

    std::tr1::shared_ptr<trrom::mock::SVD> svd(new trrom::mock::SVD);
    std::tr1::shared_ptr<trrom::mock::MatlabQR> qr(new trrom::mock::MatlabQR);
    trrom::SpectralDecompositionMng mng(svd, qr);

    mng.setDualSnapshotEnsemble(snapshots);
    mng.setDualSingularValues(singular_values);
    mng.setDualLeftSingularVectors(left_singular_vectors);
    mng.computeDualOrthonormalBasis(basis);

    trrom::SerialVector<double> data(ndofs * num_singular_values, 0.);
    trrom::Basis<double> gold(x, num_singular_values);
    data[0] = 0.511224489804523;
    data[6] = 0.252703973946664;
    data[12] = 0.492567560436969;
    data[1] = -0.407783056434706;
    data[7] = -0.441599231055225;
    data[13] = 0.464553014644157;
    data[2] = 0.325358493436934;
    data[8] = -0.216040031438741;
    data[14] = 0.404247132941033;
    data[3] = 0.542640199339754;
    data[9] = -0.231849487408929;
    data[15] = 0.104986384154719;
    data[4] = -0.303378590570133;
    data[10] = -0.313613547154724;
    data[16] = 0.417602795764322;
    data[5] = -0.282855785695668;
    data[11] = 0.736444080516773;
    data[17] = 0.439019036729722;
    gold.copy(data);
    trrom::test::checkResults(gold, basis, 1e-6);
}

TEST(SpectralDecompositionMng, state_pod)
{
    int ndofs = 6;
    int num_singular_values = 3;
    trrom::SerialVector<double> x(ndofs);
    trrom::Basis<double> basis(x, num_singular_values);
    trrom::Basis<double> snapshots(x, num_singular_values);

    trrom::SerialVector<double> singular_values(num_singular_values);

    trrom::SerialVector<double> y(num_singular_values);
    trrom::Basis<double> left_singular_vectors(y, num_singular_values);

    // Set snapshots for unit test
    snapshots.replaceGlobalValue(0, 0, 0.792207329559554);
    snapshots.replaceGlobalValue(0, 1, 0.678735154857774);
    snapshots.replaceGlobalValue(0, 2, 0.706046088019609);
    snapshots.replaceGlobalValue(1, 0, 0.959492426392903);
    snapshots.replaceGlobalValue(1, 1, 0.757740130578333);
    snapshots.replaceGlobalValue(1, 2, 0.031832846377421);
    snapshots.replaceGlobalValue(2, 0, 0.655740699156587);
    snapshots.replaceGlobalValue(2, 1, 0.743132468124916);
    snapshots.replaceGlobalValue(2, 2, 0.276922984960890);
    snapshots.replaceGlobalValue(3, 0, 0.035711678574190);
    snapshots.replaceGlobalValue(3, 1, 0.392227019534168);
    snapshots.replaceGlobalValue(3, 2, 0.046171390631154);
    snapshots.replaceGlobalValue(4, 0, 0.849129305868777);
    snapshots.replaceGlobalValue(4, 1, 0.655477890177557);
    snapshots.replaceGlobalValue(4, 2, 0.097131781235848);
    snapshots.replaceGlobalValue(5, 0, 0.933993247757551);
    snapshots.replaceGlobalValue(5, 1, 0.171186687811562);
    snapshots.replaceGlobalValue(5, 2, 0.823457828327293);
    // Set singular values for unit test
    singular_values[0] = 0.176291651111878;
    singular_values[1] = 0.695058842537319;
    singular_values[2] = 6.167248315069438;
    // Set singular vectors for unit test
    left_singular_vectors.replaceGlobalValue(0, 0, -0.655755263680298);
    left_singular_vectors.replaceGlobalValue(0, 1, 0.057661128661978);
    left_singular_vectors.replaceGlobalValue(0, 2, 0.752768376326350);
    left_singular_vectors.replaceGlobalValue(1, 0, 0.584312694702689);
    left_singular_vectors.replaceGlobalValue(1, 1, -0.592633544619941);
    left_singular_vectors.replaceGlobalValue(1, 2, 0.554404325921513);
    left_singular_vectors.replaceGlobalValue(2, 0, 0.478083370307800);
    left_singular_vectors.replaceGlobalValue(2, 1, 0.803405673388377);
    left_singular_vectors.replaceGlobalValue(2, 2, 0.354930436849960);

    std::tr1::shared_ptr<trrom::mock::SVD> svd(new trrom::mock::SVD);
    std::tr1::shared_ptr<trrom::mock::MatlabQR> qr(new trrom::mock::MatlabQR);
    trrom::SpectralDecompositionMng mng(svd, qr);

    mng.setStateSnapshotEnsemble(snapshots);
    mng.setStateSingularValues(singular_values);
    mng.setStateLeftSingularVectors(left_singular_vectors);
    mng.computeStateOrthonormalBasis(basis);

    trrom::SerialVector<double> data(ndofs * num_singular_values, 0.);
    trrom::Basis<double> gold(x, num_singular_values);
    data[0] = 0.511224489804523;
    data[6] = 0.252703973946664;
    data[12] = 0.492567560436969;
    data[1] = -0.407783056434706;
    data[7] = -0.441599231055225;
    data[13] = 0.464553014644157;
    data[2] = 0.325358493436934;
    data[8] = -0.216040031438741;
    data[14] = 0.404247132941033;
    data[3] = 0.542640199339754;
    data[9] = -0.231849487408929;
    data[15] = 0.104986384154719;
    data[4] = -0.303378590570133;
    data[10] = -0.313613547154724;
    data[16] = 0.417602795764322;
    data[5] = -0.282855785695668;
    data[11] = 0.736444080516773;
    data[17] = 0.439019036729722;
    gold.copy(data);
    trrom::test::checkResults(gold, basis, 1e-6);
}

TEST(SpectralDecompositionMng, lhs_pod)
{
    int ndofs = 6;
    int num_singular_values = 3;
    trrom::SerialVector<double> x(ndofs);
    trrom::Basis<double> basis(x, num_singular_values);
    trrom::Basis<double> snapshots(x, num_singular_values);

    trrom::SerialVector<double> singular_values(num_singular_values);

    trrom::SerialVector<double> y(num_singular_values);
    trrom::Basis<double> left_singular_vectors(y, num_singular_values);

    // Set snapshots for unit test
    snapshots.replaceGlobalValue(0, 0, 0.792207329559554);
    snapshots.replaceGlobalValue(0, 1, 0.678735154857774);
    snapshots.replaceGlobalValue(0, 2, 0.706046088019609);
    snapshots.replaceGlobalValue(1, 0, 0.959492426392903);
    snapshots.replaceGlobalValue(1, 1, 0.757740130578333);
    snapshots.replaceGlobalValue(1, 2, 0.031832846377421);
    snapshots.replaceGlobalValue(2, 0, 0.655740699156587);
    snapshots.replaceGlobalValue(2, 1, 0.743132468124916);
    snapshots.replaceGlobalValue(2, 2, 0.276922984960890);
    snapshots.replaceGlobalValue(3, 0, 0.035711678574190);
    snapshots.replaceGlobalValue(3, 1, 0.392227019534168);
    snapshots.replaceGlobalValue(3, 2, 0.046171390631154);
    snapshots.replaceGlobalValue(4, 0, 0.849129305868777);
    snapshots.replaceGlobalValue(4, 1, 0.655477890177557);
    snapshots.replaceGlobalValue(4, 2, 0.097131781235848);
    snapshots.replaceGlobalValue(5, 0, 0.933993247757551);
    snapshots.replaceGlobalValue(5, 1, 0.171186687811562);
    snapshots.replaceGlobalValue(5, 2, 0.823457828327293);
    // Set singular values for unit test
    singular_values[0] = 0.176291651111878;
    singular_values[1] = 0.695058842537319;
    singular_values[2] = 6.167248315069438;
    // Set singular vectors for unit test
    left_singular_vectors.replaceGlobalValue(0, 0, -0.655755263680298);
    left_singular_vectors.replaceGlobalValue(0, 1, 0.057661128661978);
    left_singular_vectors.replaceGlobalValue(0, 2, 0.752768376326350);
    left_singular_vectors.replaceGlobalValue(1, 0, 0.584312694702689);
    left_singular_vectors.replaceGlobalValue(1, 1, -0.592633544619941);
    left_singular_vectors.replaceGlobalValue(1, 2, 0.554404325921513);
    left_singular_vectors.replaceGlobalValue(2, 0, 0.478083370307800);
    left_singular_vectors.replaceGlobalValue(2, 1, 0.803405673388377);
    left_singular_vectors.replaceGlobalValue(2, 2, 0.354930436849960);

    std::tr1::shared_ptr<trrom::mock::SVD> svd(new trrom::mock::SVD);
    std::tr1::shared_ptr<trrom::mock::MatlabQR> qr(new trrom::mock::MatlabQR);
    trrom::SpectralDecompositionMng mng(svd, qr);

    mng.setLeftHandSideSnapshotEnsemble(snapshots);
    mng.setLeftHandSideSingularValues(singular_values);
    mng.setLeftHandSideLeftSingularVectors(left_singular_vectors);
    mng.computeLeftHandSideOrthonormalBasis(basis);

    trrom::SerialVector<double> data(ndofs * num_singular_values, 0.);
    trrom::Basis<double> gold(x, num_singular_values);
    data[0] = 0.511224489804523;
    data[6] = 0.252703973946664;
    data[12] = 0.492567560436969;
    data[1] = -0.407783056434706;
    data[7] = -0.441599231055225;
    data[13] = 0.464553014644157;
    data[2] = 0.325358493436934;
    data[8] = -0.216040031438741;
    data[14] = 0.404247132941033;
    data[3] = 0.542640199339754;
    data[9] = -0.231849487408929;
    data[15] = 0.104986384154719;
    data[4] = -0.303378590570133;
    data[10] = -0.313613547154724;
    data[16] = 0.417602795764322;
    data[5] = -0.282855785695668;
    data[11] = 0.736444080516773;
    data[17] = 0.439019036729722;
    gold.copy(data);
    trrom::test::checkResults(gold, basis, 1e-6);
}

}
