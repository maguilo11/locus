/*
 * TRROM_SimpleTest.cpp
 *
 *  Created on: Jul 22, 2016
 */

#include "gtest/gtest.h"
#include "TRROM_UtestUtils.hpp"

#include "TRROM_Data.hpp"
#include "TRROM_Basis.hpp"
#include "TRROM_Rosenbrock.hpp"
#include "TRROM_SerialVector.hpp"
#include "TRROM_MOCK_Factory.hpp"
#include "TRROM_ReducedBasis.hpp"
#include "TRROM_ReducedHessian.hpp"
#include "TRROM_ReducedBasisPDE.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_AssemblyMngTypeLP.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_ModifiedGramSchmidt.hpp"
#include "TRROM_InexactNewtonDataMng.hpp"
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
    trrom::test::checkResults(x, *matrix.vector(0));

    // number of column vectors == total number of columns
    x.fill(2.);
    matrix.insert(x);
    EXPECT_EQ(2, matrix.getNumCols());
    EXPECT_EQ(2, matrix.snapshots());
    trrom::test::checkResults(x, *matrix.vector(1));
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
    std::shared_ptr<trrom::Matrix<double> > C = B.create(3, 3);
    std::shared_ptr<trrom::Matrix<double> > gold = B.create(3, 3);
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
    std::shared_ptr<trrom::Vector<double> > z = data.create(4);
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
    int num_rows = 4;
    int num_columns = 4;
    trrom::SerialVector<double> x(num_rows);
    std::shared_ptr<trrom::Matrix<double> > A(new trrom::Basis<double>(x, num_columns));

    // basis 1
    A->replaceGlobalValue(0, 0, 10.);
    A->replaceGlobalValue(1, 0, -1.);
    A->replaceGlobalValue(2, 0, 2.);
    A->replaceGlobalValue(3, 0, 0.);
    // basis 2
    A->replaceGlobalValue(0, 1, -1.);
    A->replaceGlobalValue(1, 1, 11.);
    A->replaceGlobalValue(2, 1, -1.);
    A->replaceGlobalValue(3, 1, 3.);
    // basis 3
    A->replaceGlobalValue(0, 2, 2.);
    A->replaceGlobalValue(1, 2, -1.);
    A->replaceGlobalValue(2, 2, 10.);
    A->replaceGlobalValue(3, 2, -1.);
    // basis 4
    A->replaceGlobalValue(0, 3, 0.);
    A->replaceGlobalValue(1, 3, 3.);
    A->replaceGlobalValue(2, 3, -1.);
    A->replaceGlobalValue(3, 3, 8.);

    std::shared_ptr<trrom::mock::Factory> factory(new trrom::mock::Factory);
    trrom::ModifiedGramSchmidt method(factory);
    EXPECT_EQ(trrom::types::MODIFIED_GRAM_SCHMIDT_QR, method.type());

    std::shared_ptr<trrom::Matrix<double> > Q;
    std::shared_ptr<trrom::Matrix<double> > R;
    method.factorize(A, Q, R);

    // Check Q^t * Q = I
    trrom::Basis<double> I(x, num_columns);
    Q->gemm(true, false, 1., *Q, 0., I);
    trrom::Basis<double> mgold(x, num_columns);
    mgold.replaceGlobalValue(0, 0, 1.);
    mgold.replaceGlobalValue(1, 1, 1.);
    mgold.replaceGlobalValue(2, 2, 1.);
    mgold.replaceGlobalValue(3, 3, 1.);
    trrom::test::checkResults(mgold, I);

    // Check Q * R = A
    trrom::Basis<double> Ao(x, num_columns);
    Q->gemm(false, false, 1., *R, 0., Ao);
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
    trrom::SerialVector<double> gold(num_columns);
    gold[0] = 0.975900072948533;
    gold[1] = -0.097590007294853;
    gold[2] = 0.195180014589707;
    gold[3] = 0.;
    trrom::test::checkResults(gold, *Q->vector(0));
    // Q basis 2
    gold[0] = 0.105653526929161;
    gold[1] = 0.956798339870484;
    gold[2] = -0.049868464710564;
    gold[3] = 0.266246887861486;
    trrom::test::checkResults(gold, *Q->vector(1));
    // Q basis 3
    gold[0] = -0.186345111169031;
    gold[1] = 0.089227790175070;
    gold[2] = 0.976339450932690;
    gold[3] = -0.063837117387371;
    trrom::test::checkResults(gold, *Q->vector(2));
    // Q basis 4
    gold[0] = -0.041615855270295;
    gold[1] = -0.258943099459612;
    gold[2] = 0.078607726621668;
    gold[3] = 0.961788655135703;
    trrom::test::checkResults(gold, *Q->vector(3));

    // R column 1
    gold[0] = 10.2469507659596;
    gold[1] = 0.;
    gold[2] = 0.;
    gold[3] = 0.;
    trrom::test::checkResults(gold, *R->vector(0));
    // R column 2
    gold[0] = -2.244570167781625;
    gold[1] = 11.267737339941178;
    gold[2] = 0.;
    gold[3] = 0.;
    trrom::test::checkResults(gold, *R->vector(1));
    // R column 3
    gold[0] = 4.001190299088986;
    gold[1] = -1.510422820979288;
    gold[2] = 9.365313614201140;
    gold[3] = 0.;
    trrom::test::checkResults(gold, *R->vector(2));
    // R column 4
    gold[0] = -0.487950036474267;
    gold[1] = 5.050238587213904;
    gold[2] = -1.219353019506445;
    gold[3] = 6.838872216085120;
    trrom::test::checkResults(gold, *R->vector(3));
}

TEST(ReducedBasis, pod)
{
    int ndofs = 6;
    int nsnap = 3;
    trrom::SerialVector<double> x(ndofs);
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

    double threshold = 1;
    int num_basis_vectors = trrom::energy(threshold, singular_val);
    trrom::Basis<double> basis(x, num_basis_vectors);
    trrom::properOrthogonalDecomposition(singular_val, singular_vec, snapshots, basis);

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

TEST(SpectralDecompositionMng, allocation)
{
    trrom::SpectralDecompositionMng mng;

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
    mng.allocateDualSnapshotEnsemble(dual_snapshots);
    EXPECT_EQ(NumCols, mng.getNumDualSnapshots());

    NumCols = 4;
    trrom::Basis<double> state_snapshots(x, NumCols);
    mng.allocateStateSnapshotEnsemble(state_snapshots);
    EXPECT_EQ(NumCols, mng.getNumStateSnapshots());

    NumCols = 6;
    trrom::Basis<double> lhs_snapshots(x, NumCols);
    mng.allocateLeftHandSideSnapshotEnsemble(lhs_snapshots);
    EXPECT_EQ(NumCols, mng.getNumLeftHandSideSnapshots());
}

}
