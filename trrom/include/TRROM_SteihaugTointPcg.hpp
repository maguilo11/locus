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
    explicit SteihaugTointPcg(const std::tr1::shared_ptr<trrom::Data> & data_);
    virtual ~SteihaugTointPcg();

    void solve(const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
               const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
               const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_);

private:
    double step(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_);
    void computeStoppingTolerance(const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_);

private:
    std::tr1::shared_ptr<trrom::Vector<double> > m_NewtonStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_CauchyStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ConjugateDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_NewDescentDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_OldDescentDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_PrecTimesNewtonStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_PrecTimesConjugateDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_HessTimesConjugateDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_NewInvPrecTimesDescentDirection;
    std::tr1::shared_ptr<trrom::Vector<double> > m_OldInvPrecTimesDescentDirection;

private:
    SteihaugTointPcg(const trrom::SteihaugTointPcg &);
    trrom::SteihaugTointPcg & operator=(const trrom::SteihaugTointPcg & rhs_);
};

}

#endif /* TRROM_STEIHAUGTOINTPCG_HPP_ */
