/*
 * TRROM_TeuchosSvdTest.cpp
 *
 *  Created on: Sep 30, 2016
 *      Author: maguilo
 */

#include <stdexcept>

#include "gtest/gtest.h"

#include "TRROM_UtestUtils.hpp"
#include "TRROM_TeuchosSVD.hpp"
#include "TRROM_TeuchosSerialDenseSolver.hpp"
#include "TRROM_TeuchosSerialDenseVector.hpp"
#include "TRROM_TeuchosSerialDenseMatrix.hpp"

namespace TrromTeuchosSvdTest
{

TEST(TeuchosSerialDenseSolver, solve)
{
    int num_rows = 3;
    int num_cols = 3;
    trrom::TeuchosSerialDenseMatrix<double> A(num_rows, num_cols);
    A(0,0) = 1; A(0,1) = 0; A(0,2) = 2;
    A(1,0) = -1; A(1,1) = 5; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 3; A(2,2) = -9;
    trrom::TeuchosSerialDenseVector<double> b(num_rows);
    b[0] = 0.5376671395461;
    b[1] = 1.833885014595087;
    b[2] = -2.258846861003648;
    trrom::TeuchosSerialDenseVector<double> x(num_cols);

    trrom::TeuchosSerialDenseSolver solver;
    solver.solve(A, b, x);

    trrom::TeuchosSerialDenseVector<double> gold(num_cols);
    gold[0] = -0.184250145451618;
    gold[1] = 0.329926973828694;
    gold[2] = 0.360958642498859;

    trrom::test::checkResults(gold, x);
}

TEST(TeuchosSerialSVD, solve)
{
    int num_cols = 3;
    int num_rows = 2;
    std::tr1::shared_ptr<trrom::TeuchosSerialDenseMatrix<double> >
        A(new trrom::TeuchosSerialDenseMatrix<double>(num_rows, num_cols));
    (*A)(0, 0) = 3;
    (*A)(0, 1) = 2;
    (*A)(0, 2) = 2;
    (*A)(1, 0) = 2;
    (*A)(1, 1) = 3;
    (*A)(1, 2) = -2;

    std::tr1::shared_ptr<trrom::Vector<double> > singular_values;
    std::tr1::shared_ptr<trrom::Matrix<double> > left_singular_values;
    std::tr1::shared_ptr<trrom::Matrix<double> > right_singular_values;

    trrom::TeuchosSVD svd;
    svd.solve(A, singular_values, left_singular_values, right_singular_values);

    /* TEST SINGULAR VALUES */
    std::tr1::shared_ptr<trrom::Vector<double> > gold_singular_values = singular_values->create();
    gold_singular_values->operator [](0) = 5;
    gold_singular_values->operator [](1) = 3;
    trrom::test::checkResults(*gold_singular_values, *singular_values);

    /* TEST LEFT SINGULAR VECTORS */
    std::tr1::shared_ptr<trrom::Matrix<double> > gold_left_singular_values = left_singular_values->create();
    gold_left_singular_values->operator ()(0,0) = -1. / std::sqrt(2);
    gold_left_singular_values->operator ()(0,1) = -1. / std::sqrt(2);
    gold_left_singular_values->operator ()(1,0) = -1. / std::sqrt(2);
    gold_left_singular_values->operator ()(1,1) = 1. / std::sqrt(2);
    trrom::test::checkResults(*gold_left_singular_values, *left_singular_values);

    /* TEST RIGHT SINGULAR VECTORS */
    std::tr1::shared_ptr<trrom::Matrix<double> > gold_right_singular_values = right_singular_values->create();
    gold_right_singular_values->operator ()(0,0) = -1. / std::sqrt(2);
    gold_right_singular_values->operator ()(1,0) = -1. / std::sqrt(18);
    gold_right_singular_values->operator ()(2,0) = -2. / 3.;
    gold_right_singular_values->operator ()(0,1) = -1. / std::sqrt(2);
    gold_right_singular_values->operator ()(1,1) = 1. / std::sqrt(18);
    gold_right_singular_values->operator ()(2,1) = 2. / 3.;
    gold_right_singular_values->operator ()(0,2) = -std::numeric_limits<double>::epsilon();
    gold_right_singular_values->operator ()(1,2) = -4. / std::sqrt(18);
    gold_right_singular_values->operator ()(2,2) = 1. / 3.;
    trrom::test::checkResults(*gold_right_singular_values, *right_singular_values);
}

}
