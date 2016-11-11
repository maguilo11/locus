/*
 * TRROM_Circle.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_CIRCLE_HPP_
#define TRROM_CIRCLE_HPP_

#include "TRROM_ObjectiveTypeLP.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Circle : public trrom::ObjectiveTypeLP
{
public:
    Circle();
    virtual ~Circle();

    double value(const double & tolerance_, const trrom::Vector<double> & control_, bool & inexactness_violated_);
    void gradient(const double & tolerance_,
                  const trrom::Vector<double> & control_,
                  trrom::Vector<double> & gradient_,
                  bool & inexactness_violated_);
    void hessian(const double & tolerance_,
                 const trrom::Vector<double> & control_,
                 const trrom::Vector<double> & vector_,
                 trrom::Vector<double> & hess_times_vec_,
                 bool & inexactness_violated_);

private:
    Circle(const trrom::Circle &);
    trrom::Circle & operator=(const trrom::Circle &);
};

}

#endif /* TRROM_CIRCLE_HPP_ */
