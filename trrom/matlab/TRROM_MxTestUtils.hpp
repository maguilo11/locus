/*
 * TRROM_MxTestUtils.hpp
 *
 *  Created on: Nov 22, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXTESTUTILS_HPP_
#define TRROM_MXTESTUTILS_HPP_

#include <mex.h>
#include <cmath>
#include <string>
#include <cassert>
#include <iostream>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"

namespace trrom
{

namespace mx
{

inline void assert_test(const std::string & test_name_, bool assertion_)
{
    if(assertion_ == false)
    {
        mexPrintf("TEST NAME: %s <<FAIL>>\n", test_name_.c_str());
    }
    else
    {
        mexPrintf("TEST NAME: %s <<PASS>>\n", test_name_.c_str());
    }
}

inline bool checkResults(const trrom::Vector<double> & gold_,
                         const trrom::Vector<double> & results_,
                         double tolerance_ = 1e-6)
{
    assert(gold_.size() == results_.size());

    int length = gold_.size();
    bool did_test_pass = true;
    for(int index = 0; index < length; ++index)
    {
        double epsilon = gold_[index] - results_[index];
        if(std::abs(epsilon) > tolerance_)
        {
            did_test_pass = false;
            return (did_test_pass);
        }
    }
    return (did_test_pass);
}

inline bool checkResults(const trrom::Matrix<double> & gold_,
                         const trrom::Matrix<double> & results_,
                         double tolerance_ = 1e-6)
{
    assert(gold_.getNumRows() == results_.getNumRows());
    assert(gold_.getNumCols() == results_.getNumCols());

    bool did_test_pass = true;
    int num_rows = gold_.getNumRows();
    int num_columns = gold_.getNumCols();
    for(int row = 0; row < num_rows; ++row)
    {
        for(int column = 0; column < num_columns; ++column)
        {
            double epsilon = gold_(row, column) - results_(row, column);
            if(std::abs(epsilon) > tolerance_)
            {
                did_test_pass = false;
                return (did_test_pass);
            }
        }
    }
    return (did_test_pass);
}

inline void fill(trrom::Vector<double> & input_)
{
    const int my_length = input_.size();
    for(int index = 0; index < my_length; ++index)
    {
        input_[index] = static_cast<double>(index + 1);
    }
}

inline void fill(trrom::Matrix<double> & input_)
{
    const int num_rows = input_.getNumRows();
    const int num_columns = input_.getNumCols();
    for(int row = 0; row < num_rows; ++row)
    {
        for(int column = 0; column < num_columns; ++column)
        {
            int index = row + (num_rows * column) + 1;
            input_(row, column) = static_cast<double>(index);
        }
    }
}

inline void print(trrom::Matrix<double> & input_)
{
    const int num_rows = input_.getNumRows();
    const int num_columns = input_.getNumCols();
    for(int row = 0; row < num_rows; ++row)
    {
        for(int column = 0; column < num_columns; ++column)
        {
            std::cout << "(" << row << "," << column << ")=" << input_(row, column) << " " << std::flush;
        }
        std::cout << "\n" << std::flush;
    }
}

inline void print(trrom::Vector<double> & input_)
{
    const int my_length = input_.size();
    for(int index = 0; index < my_length; ++index)
    {
        std::cout << "(" << index << ")=" << input_[index] << "\n" << std::flush;
    }
}

}

}

#endif /* TRROM_MXTESTUTILS_HPP_ */
