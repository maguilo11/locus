/*
 * DOTk_LeftPrecConjGradDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCONJGRADDATAMNG_HPP_
#define DOTK_LEFTPRECCONJGRADDATAMNG_HPP_

#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;

template<typename ScalarType>
class Vector;

class DOTk_LeftPrecConjGradDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_LeftPrecConjGradDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    virtual ~DOTk_LeftPrecConjGradDataMng();

    void setLbfgsSecantLeftPreconditioner(size_t secant_storage_);
    void setLdfpSecantLeftPreconditioner(size_t secant_storage_);
    void setLsr1SecantLeftPreconditioner(size_t secant_storage_);
    void setSr1SecantLeftPreconditioner();
    void setBfgsSecantLeftPreconditioner();
    void setBarzilaiBorweinSecantLeftPreconditioner();

    virtual const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector() const;

private:
    std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_LeftPrecTimesResidual;

private:
    void allocate(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    DOTk_LeftPrecConjGradDataMng(const dotk::DOTk_LeftPrecConjGradDataMng &);
    dotk::DOTk_LeftPrecConjGradDataMng & operator=(const dotk::DOTk_LeftPrecConjGradDataMng &);
};

}

#endif /* DOTK_LEFTPRECCONJGRADDATAMNG_HPP_ */
