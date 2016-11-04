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

template<class Type>
class vector;

class DOTk_NocedalAndWrightEqualityNLP : public dotk::DOTk_EqualityConstraint<Real>
{
public:
    DOTk_NocedalAndWrightEqualityNLP();
    virtual ~DOTk_NocedalAndWrightEqualityNLP();

    virtual void residual(const dotk::vector<Real> & state_,
                          const dotk::vector<Real> & control_,
                          dotk::vector<Real> & residual_);
    virtual void partialDerivativeState(const dotk::vector<Real> & state_,
                                        const dotk::vector<Real> & control_,
                                        const dotk::vector<Real> & vector_,
                                        dotk::vector<Real> & output_);
    virtual void partialDerivativeControl(const dotk::vector<Real> & state_,
                                          const dotk::vector<Real> & control_,
                                          const dotk::vector<Real> & vector_,
                                          dotk::vector<Real> & output_);
    virtual void adjointPartialDerivativeState(const dotk::vector<Real> & state_,
                                               const dotk::vector<Real> & control_,
                                               const dotk::vector<Real> & dual_,
                                               dotk::vector<Real> & output_);
    virtual void adjointPartialDerivativeControl(const dotk::vector<Real> & state_,
                                                 const dotk::vector<Real> & control_,
                                                 const dotk::vector<Real> & dual_,
                                                 dotk::vector<Real> & output_);
    virtual void partialDerivativeStateState(const dotk::vector<Real> & state_,
                                             const dotk::vector<Real> & control_,
                                             const dotk::vector<Real> & dual_,
                                             const dotk::vector<Real> & vector_,
                                             dotk::vector<Real> & output_);
    virtual void partialDerivativeStateControl(const dotk::vector<Real> & state_,
                                               const dotk::vector<Real> & control_,
                                               const dotk::vector<Real> & dual_,
                                               const dotk::vector<Real> & vector_,
                                               dotk::vector<Real> & output_);
    virtual void partialDerivativeControlControl(const dotk::vector<Real> & state_,
                                                 const dotk::vector<Real> & control_,
                                                 const dotk::vector<Real> & dual_,
                                                 const dotk::vector<Real> & vector_,
                                                 dotk::vector<Real> & output_);
    virtual void partialDerivativeControlState(const dotk::vector<Real> & state_,
                                               const dotk::vector<Real> & control_,
                                               const dotk::vector<Real> & dual_,
                                               const dotk::vector<Real> & vector_,
                                               dotk::vector<Real> & output_);

private:
    DOTk_NocedalAndWrightEqualityNLP(const dotk::DOTk_NocedalAndWrightEqualityNLP&);
    dotk::DOTk_NocedalAndWrightEqualityNLP operator=(const dotk::DOTk_NocedalAndWrightEqualityNLP&);
};

}

#endif /* DOTK_NOCEDALANDWRIGHTEQUALITYNLP_HPP_ */
