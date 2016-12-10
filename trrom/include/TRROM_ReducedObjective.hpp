/*
 * TRROM_ReducedObjective.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_REDUCEDOBJECTIVE_HPP_
#define TRROM_REDUCEDOBJECTIVE_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class ReducedObjective
{
public:
    virtual ~ReducedObjective()
    {
    }

    virtual double value(const double & tolerance_,
                         const trrom::Vector<double> & state_,
                         const trrom::Vector<double> & control_,
                         bool & inexactness_violated_) = 0;
    virtual void gradient(const double & tolerance_,
                          const trrom::Vector<double> & state_,
                          const trrom::Vector<double> & control_,
                          trrom::Vector<double> & gradient_,
                          bool & inexactness_violated_) = 0;
    virtual void hessian(const double & tolerance_,
                         const trrom::Vector<double> & state_,
                         const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & hess_times_vec_,
                         bool & inexactness_violated_) = 0;
};

}

#endif /* TRROM_REDUCEDOBJECTIVE_HPP_ */
