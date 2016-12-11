/*
 * DOTk_LeftPrecGenConjResDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECGENCONJRESDATAMNG_HPP_
#define DOTK_LEFTPRECGENCONJRESDATAMNG_HPP_

#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;

template<typename ScalarType>
class Vector;

class DOTk_LeftPrecGenConjResDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_LeftPrecGenConjResDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                   size_t max_num_itr_ = 200);

    virtual ~DOTk_LeftPrecGenConjResDataMng();

    void setLbfgsSecantLeftPreconditioner(size_t secant_storage_);
    void setLdfpSecantLeftPreconditioner(size_t secant_storage_);
    void setLsr1SecantLeftPreconditioner(size_t secant_storage_);
    void setSr1SecantLeftPreconditioner();
    void setBfgsSecantLeftPreconditioner();
    void setBarzilaiBorweinSecantLeftPreconditioner();

    virtual const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector() const;

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_LeftPrecTimesResidual;
    std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;

private:
    void allocate(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    DOTk_LeftPrecGenConjResDataMng(const dotk::DOTk_LeftPrecGenConjResDataMng &);
    dotk::DOTk_LeftPrecGenConjResDataMng & operator=(const dotk::DOTk_LeftPrecGenConjResDataMng &);
};

}

#endif /* DOTK_LEFTPRECGENCONJRESDATAMNG_HPP_ */
