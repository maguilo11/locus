/*
 * DOTk_EigenUtils.hpp
 *
 *  Created on: Jul 8, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EIGENUTILS_HPP_
#define DOTK_EIGENUTILS_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

namespace eigen
{

Real rayleighQuotientMethod(const dotk::matrix<Real> & matrix_,
                            dotk::Vector<Real> & eigenvector_,
                            size_t max_num_itr_ = 10,
                            Real relative_difference_tolerance_ = 1e-6);
Real powerMethod(const dotk::matrix<Real> & matrix_,
                 dotk::Vector<Real> & eigenvector_,
                 size_t max_num_itr_ = 10,
                 Real relative_difference_tolerance_ = 1e-6);

}

}

#endif /* DOTK_EIGENUTILS_HPP_ */
