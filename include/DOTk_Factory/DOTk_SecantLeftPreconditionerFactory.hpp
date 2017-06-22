/*
 * DOTk_SecantLeftPreconditionerFactory.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SECANTLEFTPRECONDITIONERFACTORY_HPP_
#define DOTK_SECANTLEFTPRECONDITIONERFACTORY_HPP_

#include <memory>
#include <DOTk_Types.hpp>

namespace dotk
{

class DOTK_LeftPreconditioner;

template<typename ScalarType>
class Vector;

class DOTk_SecantLeftPreconditionerFactory
{
public:
    DOTk_SecantLeftPreconditionerFactory();
    ~DOTk_SecantLeftPreconditionerFactory();

    dotk::types::invhessian_t getSecantType() const;
    void buildLdfpSecantPreconditioner(size_t secant_storage_,
                                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                       std::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_);
    void buildLsr1SecantPreconditioner(size_t secant_storage_,
                                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                       std::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_);
    void buildLbfgsSecantPreconditioner(size_t secant_storage_,
                                        const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                        std::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_);
    void buildBfgsSecantPreconditioner(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                       std::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_);
    void buildSr1SecantPreconditioner(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                      std::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_);
    void buildBarzilaiBorweinSecantPreconditioner(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                                  std::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_);

private:
    void setSecantType(dotk::types::invhessian_t type_);

private:
    dotk::types::invhessian_t m_SecantType;
    dotk::types::left_prec_t m_FactoryType;

private:
    DOTk_SecantLeftPreconditionerFactory(const dotk::DOTk_SecantLeftPreconditionerFactory &);
    dotk::DOTk_SecantLeftPreconditionerFactory & operator=(const dotk::DOTk_SecantLeftPreconditionerFactory &);
};

}

#endif /* DOTK_SECANTLEFTPRECONDITIONERFACTORY_HPP_ */
