/*
 * DOTk_FirstOrderOperator.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_FIRSTORDEROPERATOR_HPP_
#define DOTK_FIRSTORDEROPERATOR_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_FirstOrderOperator
{
public:
    explicit DOTk_FirstOrderOperator(dotk::types::gradient_t type_ = dotk::types::GRADIENT_OPERATOR_DISABLED);
    virtual ~DOTk_FirstOrderOperator();

    dotk::types::gradient_t type() const;
    void checkGrad(const std::shared_ptr<dotk::Vector<Real> > & old_gradient_,
                   std::shared_ptr<dotk::Vector<Real> > & new_gradient_);

    virtual void setFiniteDiffPerturbationVec(const dotk::Vector<Real> & input_);
    virtual void gradient(const dotk::DOTk_OptimizationDataMng * const mng_);

private:
    dotk::types::gradient_t m_GradientType;

private:
    DOTk_FirstOrderOperator(const dotk::DOTk_FirstOrderOperator &);
    DOTk_FirstOrderOperator operator=(const dotk::DOTk_FirstOrderOperator &);
};

}

#endif /* DOTK_FIRSTORDEROPERATOR_HPP_ */
