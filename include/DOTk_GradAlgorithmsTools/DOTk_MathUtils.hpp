/*
 * DOTk_MathUtils.hpp
 *
 *  Created on: Oct 14, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MATHUTILS_HPP_
#define DOTK_MATHUTILS_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<class Type>
class vector;

Real norm(const std::tr1::shared_ptr<dotk::vector<Real> > & input_);
void copy(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
          const std::tr1::shared_ptr<dotk::vector<Real> > & output_);
void scale(const Real & alpha_, const std::tr1::shared_ptr<dotk::vector<Real> > & output_);
void axpy(const Real & alpha_,
          const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
          const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

Real frobeniusNorm(const std::vector< std::vector<Real> > & matrix_);
void givens(const Real & a_, const Real & b_, Real & cosine_, Real & sine_);

template<typename Type>
inline int sign(const Type & value_)
{
    if(value_ > static_cast<Type>(0))
    {
        return(1);
    }
    else
    {
        return(-1);
    }
}

}

#endif /* DOTK_MATHUTILS_HPP_ */
