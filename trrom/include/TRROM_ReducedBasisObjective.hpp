/*
 * TRROM_ReducedBasisObjective.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASISOBJECTIVE_HPP_
#define TRROM_REDUCEDBASISOBJECTIVE_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class ReducedBasisObjective
{
public:
    virtual ~ReducedBasisObjective()
    {
    }

    virtual double value(const double & tolerance_,
                         const trrom::Vector<double> & state_,
                         const trrom::Vector<double> & control_,
                         bool & inexactness_violated_) = 0;

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

    virtual void fidelity(trrom::types::fidelity_t input_) = 0;
    virtual bool checkGradientInexactness(const double & tolerance_,
                                          const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          const trrom::Vector<double> & gradient_) = 0;
};

}

#endif /* TRROM_REDUCEDBASISOBJECTIVE_HPP_ */
