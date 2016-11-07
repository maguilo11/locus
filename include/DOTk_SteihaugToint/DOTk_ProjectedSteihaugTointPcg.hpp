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

template<typename Type>
class vector;

class DOTk_ProjectedSteihaugTointPcg : public dotk::DOTk_SteihaugTointSolver
{
public:
    explicit DOTk_ProjectedSteihaugTointPcg(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_ProjectedSteihaugTointPcg();

    const std::tr1::shared_ptr<dotk::vector<Real> > & getActiveSet() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getInactiveSet() const;
    void solve(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
               const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
               const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void initialize();

    void iterate(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
               const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
               const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    Real step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
              const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    void applyVectorToHessian(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                              const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                              const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                              std::tr1::shared_ptr<dotk::vector<Real> > & output_);
    void applyVectorToPreconditioner(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                     const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                     const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                     std::tr1::shared_ptr<dotk::vector<Real> > & output_);
    void applyVectorToInvPreconditioner(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                        const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                        const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                        std::tr1::shared_ptr<dotk::vector<Real> > & output_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Residual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ActiveSet;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewtonStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CauchyStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_WorkVector;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InactiveSet;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ActiveVector;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InactiveVector;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_PrecTimesNewtonStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InvPrecTimesResidual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_PrecTimesConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_HessTimesConjugateDirection;

private:
    DOTk_ProjectedSteihaugTointPcg(const dotk::DOTk_ProjectedSteihaugTointPcg &);
    dotk::DOTk_ProjectedSteihaugTointPcg & operator=(const dotk::DOTk_ProjectedSteihaugTointPcg & rhs_);
};

}

#endif /* DOTK_PROJECTEDSTEIHAUGTOINTPCG_HPP_ */
