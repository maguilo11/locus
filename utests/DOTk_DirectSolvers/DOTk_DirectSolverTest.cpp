/*
 * DOTk_DirectSolverTest.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Types.hpp"
#include "DOTk_DirectSolver.hpp"
#include "DOTk_UpperTriangularMatrix.hpp"
#include "DOTk_UpperTriangularMatrix.cpp"
#include "DOTk_UpperTriangularDirectSolver.hpp"
#include "DOTk_LowerTriangularDirectSolver.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDirectSolverTest
{

TEST(DOTk_DirectSolver, getDirectSolverType)
{
    dotk::DOTk_DirectSolver solver(dotk::types::DIRECT_SOLVER_DISABLED);

    EXPECT_EQ(dotk::types::DIRECT_SOLVER_DISABLED, solver.getDirectSolverType());
    solver.setDirectSolverType(dotk::types::USER_DEFINED_DIRECT_SOLVER);
    EXPECT_EQ(dotk::types::USER_DEFINED_DIRECT_SOLVER, solver.getDirectSolverType());
}

TEST(DOTk_DirectSolver, getNumUnknowns)
{
    dotk::DOTk_DirectSolver solver(dotk::types::DIRECT_SOLVER_DISABLED);
    EXPECT_EQ(0, solver.getNumUnknowns());
    solver.setNumUnknowns(2);
    EXPECT_EQ(2, solver.getNumUnknowns());
}

TEST(DOTk_DirectSolver, backwardSolve)
{
    size_t num_of_unknowns = 3;
    dotk::DOTk_UpperTriangularDirectSolver solver(num_of_unknowns);
    EXPECT_EQ(dotk::types::UPPER_TRIANGULAR_DIRECT_SOLVER, solver.getDirectSolverType());

    std::shared_ptr<dotk::matrix<Real> > matrix = std::make_shared<dotk::serial::DOTk_UpperTriangularMatrix<Real>>(num_of_unknowns);
    (*matrix)(0, 0) = 1.1;
    (*matrix)(0, 1) = 4.9;
    (*matrix)(0, 2) = 7.7;
    (*matrix)(1, 1) = -12.1;
    (*matrix)(1, 2) = -22.7;
    (*matrix)(2, 2) = -1.344703226;

    std::vector<Real> rhs(num_of_unknowns, 0.);
    rhs[0] = 3.4;
    rhs[1] = 5.4;
    rhs[2] = 3.3;

    std::vector<Real> solution(num_of_unknowns, 0.);
    solver.backwardSolve(matrix, rhs, solution);

    std::vector<Real>  gold(num_of_unknowns, 0.);
    gold[0] = 1.749018783435810;
    gold[1] = 4.157641250144391;
    gold[2] = -2.454073089284014;
    dotk::gtest::checkResults(gold, solution);
}

TEST(DOTk_DirectSolver, forwardSolve)
{
    size_t num_of_unknowns = 3;
    dotk::DOTk_LowerTriangularDirectSolver solver(num_of_unknowns);
    EXPECT_EQ(dotk::types::LOWER_TRIANGULAR_DIRECT_SOLVER, solver.getDirectSolverType());

    std::vector< std::vector<Real> > matrix;
    std::vector<Real> row1(1, 0);
    row1[0] = 1.1;
    std::vector<Real> row2(2, 0);
    row2[0] = 4.9;
    row2[1] = -12.1;
    std::vector<Real> row3(3, 0);
    row3[0] = 7.7;
    row3[1] = -22.7;
    row3[2] = -1.344703226;
    matrix.push_back(row1);
    matrix.push_back(row2);
    matrix.push_back(row3);

    std::vector<Real> rhs(num_of_unknowns, 0.);
    rhs[0] = 3.4;
    rhs[1] = 5.4;
    rhs[2] = 3.3;

    std::vector<Real> solution(num_of_unknowns, 0.);
    solver.forwardSolve(matrix, rhs, solution);

    std::vector<Real>  gold(num_of_unknowns, 0.);
    gold[0] = 3.090909090909090;
    gold[1] = 0.805409466566491;
    gold[2] = 1.648843451901294;
    dotk::gtest::checkResults(gold, solution);
}

}
