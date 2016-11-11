/*
 * TRROM_ReducedObjectiveOperators.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_REDUCEDOBJECTIVEOPERATORS_HPP_
#define TRROM_REDUCEDOBJECTIVEOPERATORS_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class ReducedObjectiveOperators
{
public:
    virtual ~ReducedObjectiveOperators()
    {
    }

    virtual double value(const trrom::Vector<double> & state_,
                         const trrom::Vector<double> & control_) = 0;

    virtual bool checkObjectiveInexactness(const double & tolerance_,
                                           const double & objective_value_,
                                           const trrom::Vector<double> & state_,
                                           const trrom::Vector<double> & control_) = 0;
    virtual bool checkGradientInexactness(const double & tolerance_,
                                          const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          const trrom::Vector<double> & gradient_) = 0;
    virtual void partialDerivativeState(const trrom::Vector<double> & state_,
                                        const trrom::Vector<double> & control_,
                                        const trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeControl(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          trrom::Vector<double> & output_) = 0;

    virtual void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & vector_,
                                                 trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                             const trrom::Vector<double> & control_,
                                             const trrom::Vector<double> & vector_,
                                             trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
};

}

#endif /* TRROM_REDUCEDOBJECTIVEOPERATORS_HPP_ */
