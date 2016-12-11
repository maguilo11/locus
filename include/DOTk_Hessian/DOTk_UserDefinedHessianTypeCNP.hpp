/*
 * DOTk_UserDefinedHessianTypeCNP.hpp
 *
 *  Created on: Dec 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_USERDEFINEDHESSIANTYPECNP_HPP_
#define DOTK_USERDEFINEDHESSIANTYPECNP_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_AssemblyManager;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_UserDefinedHessianTypeCNP : public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_UserDefinedHessianTypeCNP();
    virtual ~DOTk_UserDefinedHessianTypeCNP();

    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & hessian_times_vector_);

private:
    DOTk_UserDefinedHessianTypeCNP(const dotk::DOTk_UserDefinedHessianTypeCNP &);
    dotk::DOTk_UserDefinedHessianTypeCNP & operator=(const dotk::DOTk_UserDefinedHessianTypeCNP &);
};

}

#endif /* DOTK_USERDEFINEDHESSIANTYPECNP_HPP_ */
