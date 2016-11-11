/*
 * TRROM_SteihaugTointSolver.hpp
 *
 *  Created on: Apr 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_STEIHAUGTOINTSOLVER_HPP_
#define TRROM_STEIHAUGTOINTSOLVER_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Preconditioner;
class LinearOperator;
class OptimizationDataMng;

class SteihaugTointSolver
{
public:
    SteihaugTointSolver();
    virtual ~SteihaugTointSolver();

    void setMaxNumItr(int input_);
    int getMaxNumItr() const;
    void setNumItrDone(int input_);
    int getNumItrDone() const;

    void setSolverTolerance(double input_);
    double getSolverTolerance() const;
    void setTrustRegionRadius(double input_);
    double getTrustRegionRadius() const;
    void setResidualNorm(double input_);
    double getResidualNorm() const;
    void setRelativeTolerance(double input_);
    double getRelativeTolerance() const;
    void setRelativeToleranceExponential(double input_);
    double getRelativeToleranceExponential() const;

    void setStoppingCriterion(trrom::types::solver_stop_criterion_t input_);
    trrom::types::solver_stop_criterion_t getStoppingCriterion() const;

    double computeSteihaugTointStep(const std::tr1::shared_ptr<trrom::Vector<double> > & newton_step_,
                                  const std::tr1::shared_ptr<trrom::Vector<double> > & conjugate_dir_,
                                  const std::tr1::shared_ptr<trrom::Vector<double> > & prec_times_newton_step_,
                                  const std::tr1::shared_ptr<trrom::Vector<double> > & prec_times_conjugate_dir_);
    bool invalidCurvatureDetected(const double & input_);
    bool toleranceSatisfied(const double & input_);

    virtual const std::tr1::shared_ptr<trrom::Vector<double> > & getActiveSet() const;
    virtual const std::tr1::shared_ptr<trrom::Vector<double> > & getInactiveSet() const;
    virtual void solve(const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                       const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                       const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_) = 0;

private:
    int m_MaxNumItr;
    int m_NumItrDone;

    double m_Tolerance;
    double m_ResidualNorm;
    double m_TrustRegionRadius;
    double m_RelativeTolerance;
    double m_RelativeToleranceExponential;

    trrom::types::solver_stop_criterion_t m_StoppingCriterion;

private:
    SteihaugTointSolver(const trrom::SteihaugTointSolver &);
    trrom::SteihaugTointSolver & operator=(const trrom::SteihaugTointSolver & rhs_);
};

}

#endif /* TRROM_STEIHAUGTOINTSOLVER_HPP_ */
