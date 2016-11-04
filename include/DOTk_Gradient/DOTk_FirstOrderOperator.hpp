/*
 * DOTk_FirstOrderOperator.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_FIRSTORDEROPERATOR_HPP_
#define DOTK_FIRSTORDEROPERATOR_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class T>
class vector;

class DOTk_FirstOrderOperator
{
public:
    explicit DOTk_FirstOrderOperator(dotk::types::gradient_t type_ = dotk::types::GRADIENT_OPERATOR_DISABLED);
    virtual ~DOTk_FirstOrderOperator();

    dotk::types::gradient_t type() const;
    void checkGrad(const std::tr1::shared_ptr<dotk::vector<Real> > & old_gradient_,
                   std::tr1::shared_ptr<dotk::vector<Real> > & new_gradient_);

    virtual void setFiniteDiffPerturbationVec(const dotk::vector<Real> & input_);
    virtual void gradient(const dotk::DOTk_OptimizationDataMng * const mng_);

private:
    dotk::types::gradient_t m_GradientType;

private:
    DOTk_FirstOrderOperator(const dotk::DOTk_FirstOrderOperator &);
    DOTk_FirstOrderOperator operator=(const dotk::DOTk_FirstOrderOperator &);
};

}

#endif /* DOTK_FIRSTORDEROPERATOR_HPP_ */
