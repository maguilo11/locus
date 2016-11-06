/*
 * DOTk_DescentDirectionTools.hpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DESCENTDIRECTIONTOOLS_HPP_
#define DOTK_DESCENTDIRECTIONTOOLS_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

namespace gtools
{

void getSteepestDescent(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

Real computeCosineAngle(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);

void checkDescentDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                           const std::tr1::shared_ptr<dotk::vector<Real> > & dir_,
                           Real tol_ = 1e-2);

bool didDataChanged(const std::tr1::shared_ptr<dotk::vector<Real> > & old_data_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & new_data_);

void generateRandomVector(const std::tr1::shared_ptr< dotk::vector<Real> > & input_);

template<typename Type>
Type random(Type min_, Type max_);

}

}


#endif /* DOTK_DESCENTDIRECTIONTOOLS_HPP_ */
