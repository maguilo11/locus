/*
 * DOTk_DescentDirectionTools.hpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DESCENTDIRECTIONTOOLS_HPP_
#define DOTK_DESCENTDIRECTIONTOOLS_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

namespace gtools
{

void getSteepestDescent(const std::shared_ptr<dotk::Vector<Real> > & input_,
                        const std::shared_ptr<dotk::Vector<Real> > & output_);

Real computeCosineAngle(const std::shared_ptr<dotk::Vector<Real> > & grad_,
                        const std::shared_ptr<dotk::Vector<Real> > & dir_);

void checkDescentDirection(const std::shared_ptr<dotk::Vector<Real> > & grad_,
                           const std::shared_ptr<dotk::Vector<Real> > & dir_,
                           Real tol_ = 1e-2);

bool didDataChanged(const std::shared_ptr<dotk::Vector<Real> > & old_data_,
                    const std::shared_ptr<dotk::Vector<Real> > & new_data_);

void generateRandomVector(const std::shared_ptr< dotk::Vector<Real> > & input_);

template<typename Type>
Type random(Type min_, Type max_);

}

}


#endif /* DOTK_DESCENTDIRECTIONTOOLS_HPP_ */
