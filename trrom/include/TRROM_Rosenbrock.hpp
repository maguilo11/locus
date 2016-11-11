/*
 * TRROM_Rosenbrock.hpp
 *
 *  Created on: Aug 10, 2016
 */

#ifndef TRROM_ROSENBROCK_HPP_
#define TRROM_ROSENBROCK_HPP_

#include "TRROM_ObjectiveTypeLP.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Rosenbrock : public trrom::ObjectiveTypeLP
{
public:
    Rosenbrock();
    virtual ~Rosenbrock();

    virtual double value(const double & tolerance_, const trrom::Vector<double> & control_, bool & inexactness_violated_);
    virtual void gradient(const double & tolerance_,
                          const trrom::Vector<double> & control_,
                          trrom::Vector<double> & gradient_,
                          bool & inexactness_violated_);
    virtual void hessian(const double & tolerance_,
                         const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & hess_times_vec_,
                         bool & inexactness_violated_);

private:
    Rosenbrock(const trrom::Rosenbrock &);
    trrom::Rosenbrock & operator=(const trrom::Rosenbrock &);
};

}

#endif /* TRROM_ROSENBROCK_HPP_ */
