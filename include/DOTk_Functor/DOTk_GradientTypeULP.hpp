/*
 * DOTk_GradientTypeULP.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GRADIENTTYPEULP_HPP_
#define DOTK_GRADIENTTYPEULP_HPP_

#include "DOTk_Functor.hpp"
#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_GradientTypeULP : public DOTk_Functor
{
public:
    explicit DOTk_GradientTypeULP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_);
    virtual ~DOTk_GradientTypeULP();

    virtual void operator()(const dotk::Vector<Real> & control_, dotk::Vector<Real> & gradient_);

private:
    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;

private:
    DOTk_GradientTypeULP(const dotk::DOTk_GradientTypeULP&);
    dotk::DOTk_GradientTypeULP operator=(const dotk::DOTk_GradientTypeULP&);
};

}

#endif /* DOTK_GRADIENTTYPEULP_HPP_ */
