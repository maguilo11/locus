/*
 * TRROM_ReducedSpaceAssemblyMng.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_REDUCEDSPACEASSEMBLYMNG_HPP_
#define TRROM_REDUCEDSPACEASSEMBLYMNG_HPP_

#include "TRROM_AssemblyManager.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Data;
class PDE_Constraint;
class ReducedObjectiveOperators;

class ReducedSpaceAssemblyMng : public trrom::AssemblyManager
{
public:
    ReducedSpaceAssemblyMng(const std::tr1::shared_ptr<trrom::Data> & input_,
                            const std::tr1::shared_ptr<trrom::PDE_Constraint> & pde_,
                            const std::tr1::shared_ptr<trrom::ReducedObjectiveOperators> & objective_);

    virtual ~ReducedSpaceAssemblyMng();

    virtual int getHessianCounter() const;
    virtual void updateHessianCounter();
    virtual int getGradientCounter() const;
    virtual void updateGradientCounter();
    virtual int getObjectiveCounter() const;
    virtual void updateObjectiveCounter();

    virtual double objective(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                             const double & tolerance_,
                             bool & inexactness_violated_);
    virtual void gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                          const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_,
                          const double & tolerance_,
                          bool & inexactness_violated_);
    virtual void hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                         const double & tolerance_,
                         bool & inexactness_violated_);

    int getEqualityEvaluationCounter() const;
    void updateEqualityEvaluationCounter();
    int getInverseJacobianStateCounter() const;
    void updateInverseJacobianStateCounter();
    int getAdjointInverseJacobianStateCounter() const;
    void updateInverseAdjointJacobianStateCounter();

    void useGaussNewtonHessian();

private:
    void computeHessianTimesVector(const trrom::Vector<double> & input_,
                                   const trrom::Vector<double> & trial_step_,
                                   trrom::Vector<double> & hess_times_vec_);
    void computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                  const trrom::Vector<double> & vector_,
                                  trrom::Vector<double> & hess_times_vec_);
    void computeGaussNewtonHessian(const trrom::Vector<double> & control_,
                                   const trrom::Vector<double> & vector_,
                                   trrom::Vector<double> & hess_times_vec_);

private:
    bool m_FullNewton;

    int m_HessianCounter;
    int m_GradientCounter;
    int m_ObjectiveCounter;
    int m_EqualityEvaluationCounter;
    int m_InverseJacobianStateCounter;
    int m_AdjointInverseJacobianStateCounter;

    std::tr1::shared_ptr<trrom::Vector<double> > m_Dual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_State;
    std::tr1::shared_ptr<trrom::Vector<double> > m_DeltaDual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_DeltaState;
    std::tr1::shared_ptr<trrom::Vector<double> > m_HessWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_StateWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ControlWorkVec;

    std::tr1::shared_ptr<trrom::PDE_Constraint> m_PDE;
    std::tr1::shared_ptr<trrom::ReducedObjectiveOperators> m_Objective;

private:
    ReducedSpaceAssemblyMng(const trrom::ReducedSpaceAssemblyMng &);
    trrom::ReducedSpaceAssemblyMng & operator=(const trrom::ReducedSpaceAssemblyMng &);
};

}

#endif /* TRROM_REDUCEDSPACEASSEMBLYMNG_HPP_ */
