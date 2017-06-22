/*
 * DOTk_RightPreconditionerFactory.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_RIGHTPRECONDITIONERFACTORY_HPP_
#define DOTK_RIGHTPRECONDITIONERFACTORY_HPP_

#include <string>
#include <memory>
#include <DOTk_Types.hpp>

namespace dotk
{

class DOTk_RightPreconditioner;

template<typename ScalarType>
class Vector;

class DOTk_RightPreconditionerFactory
{
public:
    explicit DOTk_RightPreconditionerFactory(dotk::types::right_prec_t type_);
    ~DOTk_RightPreconditionerFactory();

    void setWarningMsg(const std::string & msg_);
    std::string getWarningMsg() const;
    void setFactoryType(dotk::types::right_prec_t type_);
    dotk::types::right_prec_t getFactoryType() const;

    void build(const std::shared_ptr<dotk::Vector<Real> > & vec_template_,
               std::shared_ptr<dotk::DOTk_RightPreconditioner> & right_prec_);

private:
    std::string mWarningMsg;
    dotk::types::right_prec_t mFactoryType;

private:
    DOTk_RightPreconditionerFactory(const dotk::DOTk_RightPreconditionerFactory &);
    dotk::DOTk_RightPreconditionerFactory & operator=(const dotk::DOTk_RightPreconditionerFactory &);
};

}

#endif /* DOTK_RIGHTPRECONDITIONERFACTORY_HPP_ */
