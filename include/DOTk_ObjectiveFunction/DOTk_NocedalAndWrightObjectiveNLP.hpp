/*
 * DOTk_NocedalAndWrightObjectiveNLP.hpp
 *
 *  Created on: Mar 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NOCEDALANDWRIGHTOBJECTIVENLP_HPP_
#define DOTK_NOCEDALANDWRIGHTOBJECTIVENLP_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_NocedalAndWrightObjectiveNLP : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_NocedalAndWrightObjectiveNLP();
    virtual ~DOTk_NocedalAndWrightObjectiveNLP();

    virtual Real value(const dotk::Vector<Real> & state_, const dotk::Vector<Real> & control_);
    virtual void partialDerivativeState(const dotk::Vector<Real> & state_,
                                        const dotk::Vector<Real> & control_,
                                        dotk::Vector<Real> & output_);
    virtual void partialDerivativeControl(const dotk::Vector<Real> & state_,
                                          const dotk::Vector<Real> & control_,
                                          dotk::Vector<Real> & output_);
    virtual void partialDerivativeStateState(const dotk::Vector<Real> & state_,
                                             const dotk::Vector<Real> & control_,
                                             const dotk::Vector<Real> & vector_,
                                             dotk::Vector<Real> & output_);
    virtual void partialDerivativeStateControl(const dotk::Vector<Real> & state_,
                                               const dotk::Vector<Real> & control_,
                                               const dotk::Vector<Real> & vector_,
                                               dotk::Vector<Real> & output_);
    virtual void partialDerivativeControlControl(const dotk::Vector<Real> & state_,
                                                 const dotk::Vector<Real> & control_,
                                                 const dotk::Vector<Real> & vector_,
                                                 dotk::Vector<Real> & output_);
    virtual void partialDerivativeControlState(const dotk::Vector<Real> & state_,
                                               const dotk::Vector<Real> & control_,
                                               const dotk::Vector<Real> & vector_,
                                               dotk::Vector<Real> & output_);

private:
    // unimplemented
    DOTk_NocedalAndWrightObjectiveNLP(const dotk::DOTk_NocedalAndWrightObjectiveNLP&);
    dotk::DOTk_NocedalAndWrightObjectiveNLP operator=(const dotk::DOTk_NocedalAndWrightObjectiveNLP&);

};

}

#endif /* DOTK_NOCEDALANDWRIGHTOBJECTIVENLP_HPP_ */
