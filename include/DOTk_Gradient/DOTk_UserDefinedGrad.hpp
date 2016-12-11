/*
 * DOTk_UserDefinedGrad.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_USERDEFINEDGRAD_HPP_
#define DOTK_USERDEFINEDGRAD_HPP_

#include "DOTk_FirstOrderOperator.hpp"

namespace dotk
{

class DOTk_AssemblyManager;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_UserDefinedGrad : public dotk::DOTk_FirstOrderOperator
{
public:
    DOTk_UserDefinedGrad();
    virtual ~DOTk_UserDefinedGrad();

    virtual void gradient(const dotk::DOTk_OptimizationDataMng * const mng_);

private:
    DOTk_UserDefinedGrad(const dotk::DOTk_UserDefinedGrad &);
    DOTk_UserDefinedGrad operator=(const dotk::DOTk_UserDefinedGrad &);
};

}

#endif /* DOTK_USERDEFINEDGRAD_HPP_ */
