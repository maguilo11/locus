/*
 * DOTk_LeftPrecConjResDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCONJRESDATAMNG_HPP_
#define DOTK_LEFTPRECCONJRESDATAMNG_HPP_

#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;

template<typename ScalarType>
class Vector;

class DOTk_LeftPrecConjResDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_LeftPrecConjResDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    virtual ~DOTk_LeftPrecConjResDataMng();

    void setLbfgsSecantLeftPreconditioner(size_t secant_storage_);
    void setLdfpSecantLeftPreconditioner(size_t secant_storage_);
    void setLsr1SecantLeftPreconditioner(size_t secant_storage_);
    void setSr1SecantLeftPreconditioner();
    void setBfgsSecantLeftPreconditioner();
    void setBarzilaiBorweinSecantLeftPreconditioner();

    virtual const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector() const;

private:
    std::shared_ptr<dotk::Vector<Real> > m_LeftPrecTimesResidual;

    std::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;

private:
    void allocate(const std::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    DOTk_LeftPrecConjResDataMng(const dotk::DOTk_LeftPrecConjResDataMng &);
    dotk::DOTk_LeftPrecConjResDataMng & operator=(const dotk::DOTk_LeftPrecConjResDataMng &);
};

}

#endif /* DOTK_LEFTPRECCONJRESDATAMNG_HPP_ */
