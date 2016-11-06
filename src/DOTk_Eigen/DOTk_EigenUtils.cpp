/*
 * DOTk_EigenUtils.cpp
 *
 *  Created on: Jul 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_EigenUtils.hpp"

namespace dotk
{

namespace eigen
{

Real powerMethod(const dotk::matrix<Real> & matrix_,
                 dotk::vector<Real> & eigenvector_,
                 size_t max_num_itr_,
                 Real relative_difference_tolerance_)
{
    Real relative_difference, new_eigenvalue;
    Real old_eigenvalue = std::numeric_limits<Real>::max();
    std::tr1::shared_ptr<dotk::vector<Real> > old_eigenvector = eigenvector_.clone();
    for(size_t itr = 0; itr < max_num_itr_; ++itr)
    {
        old_eigenvector->copy(eigenvector_);
        eigenvector_.fill(0.);
        matrix_.matVec(*old_eigenvector, eigenvector_);
        new_eigenvalue = eigenvector_.max() / old_eigenvector->max();
        relative_difference = std::abs(new_eigenvalue - old_eigenvalue);
        if(relative_difference < relative_difference_tolerance_)
        {
            break;
        }
        old_eigenvalue = new_eigenvalue;
    }
    Real scaling = static_cast<Real>(1.) / eigenvector_.norm();
    eigenvector_.scale(scaling);
    return (new_eigenvalue);
}

Real rayleighQuotientMethod(const dotk::matrix<Real> & matrix_,
                            dotk::vector<Real> & eigenvector_,
                            size_t max_num_itr_,
                            Real relative_difference_tolerance_)
{
    Real old_eigenvalue = std::numeric_limits<Real>::max();
    Real relative_difference, new_eigenvalue, old_eigenvector_norm;
    std::tr1::shared_ptr<dotk::vector<Real> > old_eigenvector = eigenvector_.clone();
    for(size_t itr = 0; itr < max_num_itr_; ++itr)
    {
        old_eigenvector->copy(eigenvector_);
        eigenvector_.fill(0.);
        old_eigenvector_norm = old_eigenvector->norm();
        matrix_.matVec(*old_eigenvector, eigenvector_);
        new_eigenvalue = old_eigenvector->dot(eigenvector_) / (old_eigenvector_norm * old_eigenvector_norm);
        relative_difference = std::abs(new_eigenvalue - old_eigenvalue);
        if(relative_difference < relative_difference_tolerance_)
        {
            break;
        }
        old_eigenvalue = new_eigenvalue;
    }
    Real scaling = static_cast<Real>(1.) / eigenvector_.norm();
    eigenvector_.scale(scaling);
    return (new_eigenvalue);
}

}

}
