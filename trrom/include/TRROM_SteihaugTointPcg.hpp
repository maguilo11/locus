/*
 * TRROM_SteihaugTointPcg.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_STEIHAUGTOINTPCG_HPP_
#define TRROM_STEIHAUGTOINTPCG_HPP_

#include "TRROM_SteihaugTointSolver.hpp"

namespace trrom
{

class Data;
class Preconditioner;
class LinearOperator;
class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class SteihaugTointPcg : public trrom::SteihaugTointSolver
{
public:
    explicit SteihaugTointPcg(const std::shared_ptr<trrom::Data> & data_);
    virtual ~SteihaugTointPcg();

    void solve(const std::shared_ptr<trrom::Preconditioner> & preconditioner_,
               const std::shared_ptr<trrom::LinearOperator> & linear_operator_,
               const std::shared_ptr<trrom::OptimizationDataMng> & mng_);

private:
    double step(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                const std::shared_ptr<trrom::Preconditioner> & preconditioner_);
    void computeStoppingTolerance(const std::shared_ptr<trrom::Vector<double> > & gradient_);

private:
    std::shared_ptr<trrom::Vector<double> > m_NewtonStep;
    std::shared_ptr<trrom::Vector<double> > m_CauchyStep;
    std::shared_ptr<trrom::Vector<double> > m_ConjugateDirection;
    std::shared_ptr<trrom::Vector<double> > m_NewDescentDirection;
    std::shared_ptr<trrom::Vector<double> > m_OldDescentDirection;
    std::shared_ptr<trrom::Vector<double> > m_PrecTimesNewtonStep;
    std::shared_ptr<trrom::Vector<double> > m_PrecTimesConjugateDirection;
    std::shared_ptr<trrom::Vector<double> > m_HessTimesConjugateDirection;
    std::shared_ptr<trrom::Vector<double> > m_NewInvPrecTimesDescentDirection;
    std::shared_ptr<trrom::Vector<double> > m_OldInvPrecTimesDescentDirection;

private:
    SteihaugTointPcg(const trrom::SteihaugTointPcg &);
    trrom::SteihaugTointPcg & operator=(const trrom::SteihaugTointPcg & rhs_);
};

}

#endif /* TRROM_STEIHAUGTOINTPCG_HPP_ */
