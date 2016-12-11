/*
 * DOTk_Daniels.hpp
 *
 *  Created on: Jul 6, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DANIELS_HPP_
#define DOTK_DANIELS_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_LinearOperator;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_Daniels : public dotk::DOTk_DescentDirection
{
public:
    explicit DOTk_Daniels(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_);
    virtual ~DOTk_Daniels();

    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_Hessian;

private:
    DOTk_Daniels(const dotk::DOTk_Daniels &);
    dotk::DOTk_Daniels & operator=(const dotk::DOTk_Daniels &);
};

}

#endif /* DOTK_DANIELS_HPP_ */
