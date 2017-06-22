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

template<typename ScalarType>
class Vector;

class DOTk_SteihaugTointPcg : public dotk::DOTk_SteihaugTointSolver
{
public:
    explicit DOTk_SteihaugTointPcg(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_SteihaugTointPcg();

    void solve(const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
               const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
               const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real step(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
              const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    void computeStoppingTolerance(const std::shared_ptr<dotk::Vector<Real> > & gradient_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_NewtonStep;
    std::shared_ptr<dotk::Vector<Real> > m_CauchyStep;
    std::shared_ptr<dotk::Vector<Real> > m_ConjugateDirection;
    std::shared_ptr<dotk::Vector<Real> > m_NewDescentDirection;
    std::shared_ptr<dotk::Vector<Real> > m_OldDescentDirection;
    std::shared_ptr<dotk::Vector<Real> > m_PrecTimesNewtonStep;
    std::shared_ptr<dotk::Vector<Real> > m_PrecTimesConjugateDirection;
    std::shared_ptr<dotk::Vector<Real> > m_HessTimesConjugateDirection;
    std::shared_ptr<dotk::Vector<Real> > m_NewInvPrecTimesDescentDirection;
    std::shared_ptr<dotk::Vector<Real> > m_OldInvPrecTimesDescentDirection;

private:
    DOTk_SteihaugTointPcg(const dotk::DOTk_SteihaugTointPcg &);
    dotk::DOTk_SteihaugTointPcg & operator=(const dotk::DOTk_SteihaugTointPcg & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTPCG_HPP_ */
