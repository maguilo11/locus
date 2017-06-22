/*
 * DOTk_UserDefinedHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_USERDEFINEDHESSIAN_HPP_
#define DOTK_USERDEFINEDHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_UserDefinedHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_UserDefinedHessian();
    virtual ~DOTk_UserDefinedHessian();

    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & hessian_times_vector_);

private:
    DOTk_UserDefinedHessian(const dotk::DOTk_UserDefinedHessian &);
    dotk::DOTk_UserDefinedHessian & operator=(const dotk::DOTk_UserDefinedHessian &);
};

}

#endif /* DOTK_USERDEFINEDHESSIAN_HPP_ */
