/*
 * DOTk_DescentDirectionTools.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>
#include <cstdlib>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DescentDirectionTools.hpp"

namespace dotk
{

namespace gtools
{

void getSteepestDescent(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_,
                        const std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    output_->copy(*input_);
    output_->scale(static_cast<Real>(-1.));
}

Real computeCosineAngle(const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                        const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real norm_dir = dir_->norm();
    Real grad_dot_dir = grad_->dot(*dir_);
    Real norm_grad = grad_->norm();
    Real value = grad_dot_dir / (norm_dir * norm_grad);
    value = std::abs(value);
    return (value);
}

void checkDescentDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                           const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_,
                           Real tol_)
{
    Real cosine_angle = dotk::gtools::computeCosineAngle(grad_, dir_);
    if(cosine_angle < tol_)
    {
        dotk::gtools::getSteepestDescent(grad_, dir_);
    }
    else if(std::isnan(cosine_angle))
    {
        dotk::gtools::getSteepestDescent(grad_, dir_);
    }
    else if(std::isinf(cosine_angle))
    {
        dotk::gtools::getSteepestDescent(grad_, dir_);
    }
}

bool didDataChanged(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_data_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & new_data_)
{
    Real dot_old_minus_dot_new = std::abs(old_data_->dot(*old_data_) - new_data_->dot(*new_data_));
    bool did_primal_changed = dot_old_minus_dot_new > std::numeric_limits<Real>::min() ? true: false;
    return (did_primal_changed);
}

void generateRandomVector(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_)
{
    size_t num_entries = input_->size();
    for(size_t i = 0; i < num_entries; ++ i)
    {
        (*input_)[i] = (static_cast<Real>(rand())) / RAND_MAX;
    }
}

template<typename Type>
Type random(Type min_, Type max_)
{
    int range = max_ - min_ + 1;
    int remainder = RAND_MAX % range;
    int random_number;
    do
    {
        random_number = rand();
    }
    while(random_number >= RAND_MAX - remainder);
    Type value = min_ + random_number % range;
    return (value);
}

}

}
