/*
 * DOTk_GtestDOTkVecTools.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <omp.h>
#include "DOTk_SerialVector.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace dotk
{

namespace gtest
{

std::shared_ptr<dotk::Vector<Real> > allocateControl()
{
    size_t num_controls = 2;
    std::vector<Real> data(num_controls, 2.);
    std::shared_ptr<dotk::Vector<Real> > vector = std::make_shared<dotk::StdVector<Real>>(data);
    return (vector);
}

std::shared_ptr<dotk::Vector<Real> > allocateData(size_t dim_, Real value_)
{
    std::shared_ptr<dotk::Vector<Real> > vector = std::make_shared<dotk::StdVector<Real>>(dim_, value_);
    return (vector);
}

void checkResults(const dotk::Vector<Real>& results_, const dotk::Vector<Real>& gold_, Real tol_)
{
    assert(results_.size() == gold_.size());
    for(size_t i = 0; i < results_.size(); ++i)
    {
        EXPECT_NEAR(gold_[i], results_[i], tol_);
    }
}

void checkResults(const std::vector<Real> & results_, const std::vector<Real> & gold_, Real tol_)
{
    assert(results_.size() == gold_.size());
    for(size_t tIndex = 0; tIndex < results_.size(); ++tIndex)
    {
        EXPECT_NEAR(gold_.operator [](tIndex), results_.operator [](tIndex), tol_);
    }
}

void checkResults(const size_t & num_gold_values,
                  const Real* gold_,
                  const dotk::Vector<Real> & results_,
                  int thread_count_,
                  Real tol_)
{
    size_t num_result_values = results_.size();
    assert(num_gold_values == num_result_values);
    size_t index;

# pragma omp parallel num_threads(thread_count_) \
    default( none ) \
    shared ( num_result_values, tol_, gold_, results_ ) \
    private ( index )

# pragma omp for
    for(index = 0; index < results_.size(); ++ index)
    {
        EXPECT_NEAR(gold_[index], results_[index], tol_);
    }
}

void checkResults(const size_t & num_gold_values,
                  const Real* gold_,
                  const size_t & num_result_values,
                  const Real* results_,
                  int thread_count_,
                  Real tol_)
{
    assert(num_gold_values == num_result_values);

    size_t index;

# pragma omp parallel num_threads(thread_count_) \
    default( none ) \
    shared ( num_result_values, tol_, gold_, results_ ) \
    private ( index )

# pragma omp for
    for(index = 0; index < num_result_values; ++ index)
    {
        EXPECT_NEAR(gold_[index], results_[index], tol_);
    }
}

void checkResults(const dotk::Vector<Real> & gold_, const dotk::Vector<Real> & results_, int thread_count_, Real tol_)
{
    size_t i;
    size_t dim = gold_.size();
    assert(dim == results_.size());

# pragma omp parallel num_threads(thread_count_) \
    default( none ) \
    shared ( dim, tol_, gold_, results_ ) \
    private ( i )

# pragma omp for
    for(i = 0; i < dim; ++ i)
    {
        EXPECT_NEAR(gold_[i], results_[i], tol_);
    }
}

}

}
