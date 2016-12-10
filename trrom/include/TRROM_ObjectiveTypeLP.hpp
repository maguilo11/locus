/*
 * TRROM_ObjectiveTypeLP.hpp
 *
 *  Created on: Aug 10, 2016
 */

#ifndef TRROM_OBJECTIVETYPELP_HPP_
#define TRROM_OBJECTIVETYPELP_HPP_

#include "TRROM_Vector.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class ObjectiveTypeLP
{
public:
    virtual ~ObjectiveTypeLP()
    {
    }

    virtual double value(const double & tolerance_,
                         const trrom::Vector<double> & control_,
                         bool & inexactness_violated_) = 0;
    virtual void gradient(const double & tolerance_,
                          const trrom::Vector<double> & control_,
                          trrom::Vector<double> & gradient_,
                          bool & inexactness_violated_) = 0;
    virtual void hessian(const double & tolerance_,
                         const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & hess_times_vec_,
                         bool & inexactness_violated_) = 0;
};

}

#endif /* TRROM_OBJECTIVETYPELP_HPP_ */
