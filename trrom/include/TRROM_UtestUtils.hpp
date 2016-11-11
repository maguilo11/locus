/*
 * TRROM_UtestUtils.cpp
 *
 *  Created on: Sep 30, 2016
 *      Author: maguilo
 */

#include "gtest/gtest.h"

#include <cassert>
#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"

namespace trrom
{

namespace test
{

template<typename ScalarType>
void checkResults(const trrom::Vector<ScalarType> & gold_,
                  const trrom::Vector<ScalarType> & result_,
                  ScalarType tolerance_ = 1e-8,
                  int thread_count_ = 1)
{
    assert(gold_.size() == result_.size());

    int num_elements = result_.size();
    for(int index = 0; index < num_elements; ++index)
    {
        EXPECT_NEAR(gold_[index], result_[index], tolerance_);
    }
}

template<typename ScalarType>
void checkResults(const trrom::Matrix<ScalarType> & gold_, const trrom::Matrix<ScalarType> & result_, ScalarType tolerance_ = 1e-8)
{
    assert(gold_.numCols() == result_.numCols());
    assert(gold_.numRows() == result_.numRows());
    for(int column = 0; column < gold_.numCols(); ++column)
    {
        for(int row = 0; row < gold_.numRows(); ++row)
        {
            EXPECT_NEAR(gold_(row, column), result_(row, column), tolerance_);
        }
    }
}

}

}
