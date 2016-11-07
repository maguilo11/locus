/*
 * DOTk_SteihaugTointPcg.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTPCG_HPP_
#define DOTK_STEIHAUGTOINTPCG_HPP_

#include "DOTk_SteihaugTointSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_Preconditioner;
class DOTk_LinearOperator;
class DOTk_OptimizationDataMng;

template<typename Type>
class vector;

class DOTk_SteihaugTointPcg : public dotk::DOTk_SteihaugTointSolver
{
public:
    explicit DOTk_SteihaugTointPcg(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_SteihaugTointPcg();

    void solve(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
               const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
               const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
              const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    void computeStoppingTolerance(const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewtonStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CauchyStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewDescentDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OldDescentDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_PrecTimesNewtonStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_PrecTimesConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_HessTimesConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewInvPrecTimesDescentDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OldInvPrecTimesDescentDirection;

private:
    DOTk_SteihaugTointPcg(const dotk::DOTk_SteihaugTointPcg &);
    dotk::DOTk_SteihaugTointPcg & operator=(const dotk::DOTk_SteihaugTointPcg & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTPCG_HPP_ */
