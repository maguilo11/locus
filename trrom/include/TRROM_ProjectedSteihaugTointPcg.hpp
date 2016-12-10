/*
 * TRROM_ProjectedSteihaugTointPcg.hpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_PROJECTEDSTEIHAUGTOINTPCG_HPP_
#define TRROM_PROJECTEDSTEIHAUGTOINTPCG_HPP_

#include "TRROM_SteihaugTointSolver.hpp"

namespace trrom
{

class Data;
class Preconditioner;
class LinearOperator;
class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class ProjectedSteihaugTointPcg : public trrom::SteihaugTointSolver
{
public:
    explicit ProjectedSteihaugTointPcg(const std::tr1::shared_ptr<trrom::Data> & data_);
    virtual ~ProjectedSteihaugTointPcg();

    const std::tr1::shared_ptr<trrom::Vector<double> > & getActiveSet() const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & getInactiveSet() const;
    void solve(const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
               const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
               const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_);

private:
    void initialize();

    void iterate(const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                 const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                 const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_);
    double step(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_);
    void applyVectorToHessian(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                              const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                              const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                              std::tr1::shared_ptr<trrom::Vector<double> > & output_);
    void applyVectorToPreconditioner(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                     const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                                     const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                     std::tr1::shared_ptr<trrom::Vector<double> > & output_);
    void applyVectorToInvPreconditioner(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                        const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                                        const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                        std::tr1::shared_ptr<trrom::Vector<double> > & output_);

private:
    std::tr1::shared_ptr<trrom::Vector<double> > m_Residual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ActiveSet;
    std::tr1::shared_ptr<trrom::Vector<double> > m_NewtonStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_CauchyStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_WorkVector;
    std::tr1::shared_ptr<trrom::Vector<double> > m_InactiveSet;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ActiveVector;
    std::tr1::shared_ptr<trrom::Vector<double> > m_InactiveVector;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ConjugateDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_PrecTimesNewtonStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_InvPrecTimesResidual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_PrecTimesConjugateDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_HessTimesConjugateDirection;

private:
    ProjectedSteihaugTointPcg(const trrom::ProjectedSteihaugTointPcg &);
    trrom::ProjectedSteihaugTointPcg & operator=(const trrom::ProjectedSteihaugTointPcg & rhs_);
};

}

#endif /* TRROM_PROJECTEDSTEIHAUGTOINTPCG_HPP_ */
