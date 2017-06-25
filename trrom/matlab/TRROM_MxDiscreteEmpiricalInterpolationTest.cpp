/*
 * TRROM_MxDiscreteEmpiricalInterpolationTest.cpp
 *
 *  Created on: Dec 7, 2016
 *      Author: maguilo
 */

#include <string>
#include <memory>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxDirectSolver.hpp"
#include "TRROM_MxLinearAlgebraFactory.hpp"
#include "TRROM_DiscreteEmpiricalInterpolation.hpp"

namespace trrom
{

namespace mx
{

inline void setDEIMTestData(std::shared_ptr<trrom::Matrix<double> > & basis_)
{
    int num_rows = 10;
    int num_columns = 3;
    basis_.reset(new trrom::MxMatrix(num_rows, num_columns));
    (*basis_)(0, 0) = -0.305337446414953;
    (*basis_)(0, 1) = 0.149929755643348;
    (*basis_)(0, 2) = 0.454738659588238;
    (*basis_)(1, 0) = -0.161337182800233;
    (*basis_)(1, 1) = 0.297250343025266;
    (*basis_)(1, 2) = -0.0352937566808501;
    (*basis_)(2, 0) = -0.406496477165708;
    (*basis_)(2, 1) = -0.306927516771071;
    (*basis_)(2, 2) = -0.626176494523147;
    (*basis_)(3, 0) = -0.300220545037820;
    (*basis_)(3, 1) = -0.567414531313143;
    (*basis_)(3, 2) = 0.387289844252935;
    (*basis_)(4, 0) = -0.138015061324149;
    (*basis_)(4, 1) = -0.340273485747276;
    (*basis_)(4, 2) = 0.0637131106348886;
    (*basis_)(5, 0) = -0.168893004368106;
    (*basis_)(5, 1) = -0.272025188535749;
    (*basis_)(5, 2) = -0.0409104638371521;
    (*basis_)(6, 0) = -0.400119129029725;
    (*basis_)(6, 1) = 0.459345783984302;
    (*basis_)(6, 2) = 0.146343369003984;
    (*basis_)(7, 0) = -0.543617952688073;
    (*basis_)(7, 1) = 0.0322279907939728;
    (*basis_)(7, 2) = 0.0986045685918870;
    (*basis_)(8, 0) = -0.327568742239719;
    (*basis_)(8, 1) = 0.241999491250776;
    (*basis_)(8, 2) = -0.173257748931569;
    (*basis_)(9, 0) = -0.121973418343343;
    (*basis_)(9, 1) = 0.112315877446613;
    (*basis_)(9, 2) = -0.427768815809286;
}

inline void setDEIMTestGold(std::shared_ptr<trrom::Matrix<double> > & binary_matrix,
                            std::shared_ptr<trrom::Vector<double> > & active_indices_)
{
    const int num_rows = 10;
    const int num_columns = 3;
    binary_matrix.reset(new trrom::MxMatrix(num_rows, num_columns));
    (*binary_matrix)(7, 0) = 1;
    (*binary_matrix)(3, 1) = 1;
    (*binary_matrix)(2, 2) = 1;

    active_indices_.reset(new trrom::MxVector(num_columns));
    (*active_indices_)[0] = 7;
    (*active_indices_)[1] = 3;
    (*active_indices_)[2] = 2;
}

}

}

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR DEIM ALGORITHM\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 0 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES NO INPUTS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    std::shared_ptr<trrom::MxDirectSolver> solver(new trrom::MxDirectSolver);
    std::shared_ptr<trrom::MxLinearAlgebraFactory> factory(new trrom::MxLinearAlgebraFactory);
    trrom::DiscreteEmpiricalInterpolation deim(solver, factory);

   // **** TEST 1: apply ****
    std::shared_ptr<trrom::Matrix<double> > basis;
    std::shared_ptr<trrom::Matrix<double> > binary_matrix;
    std::shared_ptr<trrom::Vector<double> > active_indices;
    trrom::mx::setDEIMTestData(basis);
    binary_matrix = basis->create();
    deim.apply(basis, binary_matrix, active_indices);

    // SET GOLD VALUES
    std::shared_ptr<trrom::Matrix<double> > gold_binary_matrix;
    std::shared_ptr<trrom::Vector<double> > gold_active_indices;
    trrom::mx::setDEIMTestGold(gold_binary_matrix, gold_active_indices);

    // ASSERT TEST 1 RESULTS
    msg.assign("binary matrix");
    bool did_test_pass = trrom::mx::checkResults(*gold_binary_matrix, *binary_matrix);
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("active indices");
    did_test_pass = trrom::mx::checkResults(*gold_active_indices, *active_indices);
    trrom::mx::assert_test(msg, did_test_pass);
}
