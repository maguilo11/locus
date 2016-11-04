/*
 * DOTk_MathUtils.cpp
 *
 *  Created on: Oct 14, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>
#include "vector.hpp"
#include "DOTk_MathUtils.hpp"

namespace dotk
{

Real norm(const std::tr1::shared_ptr<dotk::vector<Real> > & input_)
{
    Real output = input_->control()->dot(*input_->control());
    if(input_->state().use_count() > 0)
    {
        assert(input_->state().use_count() > 0);
        output += input_->state()->dot(*input_->state());
    }
    output = std::sqrt(output);
    return (output);
}

void copy(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
          const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    output_->control()->copy(*input_->control());
    if(input_->state().use_count() > 0)
    {
        assert(input_->state().use_count() > 0);
        output_->state()->copy(*input_->state());
    }
}

void scale(const Real & alpha_, const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    output_->control()->scale(alpha_);
    if(output_->state().use_count() > 0)
    {
        output_->state()->scale(alpha_);
    }
}

void axpy(const Real & alpha_,
          const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
          const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    output_->control()->axpy(alpha_, *input_->control());
    if(input_->state().use_count() > 0)
    {
        assert(input_->state().use_count() > 0);
        output_->state()->axpy(alpha_, *input_->state());
    }
}

Real frobeniusNorm(const std::vector< std::vector<Real> > & matrix_)
{
    Real value = 0.;
    size_t nrows = matrix_.size();
    size_t ncols = matrix_[0].size();

    for(size_t row = 0; row < nrows; ++row)
    {
        for(size_t col = 0; col < ncols; ++col)
        {
            value += matrix_[row][col] * matrix_[row][col];
        }
    }

    value = std::sqrt(value);

    return (value);
}

void givens(const Real & a_, const Real & b_, Real & cosine_, Real & sine_)
{
    if(b_ == static_cast<Real>(0.))
    {
        cosine_ = static_cast<Real>(1.);
        sine_ = static_cast<Real>(0.);
    }
    else if(std::abs(b_) > std::abs(a_))
    {
        Real a_over_b = a_ / b_;
        sine_ = static_cast<Real>(1.) / sqrt(static_cast<Real>(1.) + a_over_b * a_over_b);
        cosine_ = a_over_b * sine_;
    }
    else
    {
        Real b_over_a = b_ / a_;
        cosine_ = static_cast<Real>(1.) / sqrt(static_cast<Real>(1.) + b_over_a * b_over_a);
        sine_ = b_over_a * cosine_;
    }
}

}
