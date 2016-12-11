/*
 * DOTk_NocedalAndWrightEqualityNLP.hpp
 *
 *  Created on: Mar 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NOCEDALANDWRIGHTEQUALITYNLP_HPP_
#define DOTK_NOCEDALANDWRIGHTEQUALITYNLP_HPP_

#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_NocedalAndWrightEqualityNLP : public dotk::DOTk_EqualityConstraint<Real>
{
public:
    DOTk_NocedalAndWrightEqualityNLP();
    virtual ~DOTk_NocedalAndWrightEqualityNLP();

    virtual void residual(const dotk::Vector<Real> & state_,
                          const dotk::Vector<Real> & control_,
                          dotk::Vector<Real> & residual_);
    virtual void partialDerivativeState(const dotk::Vector<Real> & state_,
                                        const dotk::Vector<Real> & control_,
                                        const dotk::Vector<Real> & vector_,
                                        dotk::Vector<Real> & output_);
    virtual void partialDerivativeControl(const dotk::Vector<Real> & state_,
                                          const dotk::Vector<Real> & control_,
                                          const dotk::Vector<Real> & vector_,
                                          dotk::Vector<Real> & output_);
    virtual void adjointPartialDerivativeState(const dotk::Vector<Real> & state_,
                                               const dotk::Vector<Real> & control_,
                                               const dotk::Vector<Real> & dual_,
                                               dotk::Vector<Real> & output_);
    virtual void adjointPartialDerivativeControl(const dotk::Vector<Real> & state_,
                                                 const dotk::Vector<Real> & control_,
                                                 const dotk::Vector<Real> & dual_,
                                                 dotk::Vector<Real> & output_);
    virtual void partialDerivativeStateState(const dotk::Vector<Real> & state_,
                                             const dotk::Vector<Real> & control_,
                                             const dotk::Vector<Real> & dual_,
                                             const dotk::Vector<Real> & vector_,
                                             dotk::Vector<Real> & output_);
    virtual void partialDerivativeStateControl(const dotk::Vector<Real> & state_,
                                               const dotk::Vector<Real> & control_,
                                               const dotk::Vector<Real> & dual_,
                                               const dotk::Vector<Real> & vector_,
                                               dotk::Vector<Real> & output_);
    virtual void partialDerivativeControlControl(const dotk::Vector<Real> & state_,
                                                 const dotk::Vector<Real> & control_,
                                                 const dotk::Vector<Real> & dual_,
                                                 const dotk::Vector<Real> & vector_,
                                                 dotk::Vector<Real> & output_);
    virtual void partialDerivativeControlState(const dotk::Vector<Real> & state_,
                                               const dotk::Vector<Real> & control_,
                                               const dotk::Vector<Real> & dual_,
                                               const dotk::Vector<Real> & vector_,
                                               dotk::Vector<Real> & output_);

private:
    DOTk_NocedalAndWrightEqualityNLP(const dotk::DOTk_NocedalAndWrightEqualityNLP&);
    dotk::DOTk_NocedalAndWrightEqualityNLP operator=(const dotk::DOTk_NocedalAndWrightEqualityNLP&);
};

}

#endif /* DOTK_NOCEDALANDWRIGHTEQUALITYNLP_HPP_ */
