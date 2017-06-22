/*
 * DOTk_ProjectedSteihaugTointPcg.hpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PROJECTEDSTEIHAUGTOINTPCG_HPP_
#define DOTK_PROJECTEDSTEIHAUGTOINTPCG_HPP_

#include "DOTk_SteihaugTointSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_Preconditioner;
class DOTk_LinearOperator;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_ProjectedSteihaugTointPcg : public dotk::DOTk_SteihaugTointSolver
{
public:
    explicit DOTk_ProjectedSteihaugTointPcg(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_ProjectedSteihaugTointPcg();

    const std::shared_ptr<dotk::Vector<Real> > & getActiveSet() const;
    const std::shared_ptr<dotk::Vector<Real> > & getInactiveSet() const;
    void solve(const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
               const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
               const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void initialize();

    void iterate(const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
               const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
               const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    Real step(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
              const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    void applyVectorToHessian(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                              const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                              const std::shared_ptr<dotk::Vector<Real> > & vector_,
                              std::shared_ptr<dotk::Vector<Real> > & output_);
    void applyVectorToPreconditioner(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                     const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                     const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                     std::shared_ptr<dotk::Vector<Real> > & output_);
    void applyVectorToInvPreconditioner(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                        const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                        const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                        std::shared_ptr<dotk::Vector<Real> > & output_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_Residual;
    std::shared_ptr<dotk::Vector<Real> > m_ActiveSet;
    std::shared_ptr<dotk::Vector<Real> > m_NewtonStep;
    std::shared_ptr<dotk::Vector<Real> > m_CauchyStep;
    std::shared_ptr<dotk::Vector<Real> > m_WorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_InactiveSet;
    std::shared_ptr<dotk::Vector<Real> > m_ActiveVector;
    std::shared_ptr<dotk::Vector<Real> > m_InactiveVector;
    std::shared_ptr<dotk::Vector<Real> > m_ConjugateDirection;
    std::shared_ptr<dotk::Vector<Real> > m_PrecTimesNewtonStep;
    std::shared_ptr<dotk::Vector<Real> > m_InvPrecTimesResidual;
    std::shared_ptr<dotk::Vector<Real> > m_PrecTimesConjugateDirection;
    std::shared_ptr<dotk::Vector<Real> > m_HessTimesConjugateDirection;

private:
    DOTk_ProjectedSteihaugTointPcg(const dotk::DOTk_ProjectedSteihaugTointPcg &);
    dotk::DOTk_ProjectedSteihaugTointPcg & operator=(const dotk::DOTk_ProjectedSteihaugTointPcg & rhs_);
};

}

#endif /* DOTK_PROJECTEDSTEIHAUGTOINTPCG_HPP_ */
