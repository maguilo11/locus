/*
 * TRROM_ReducedBasisGaussNewtonAssemblyMng.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASISASSEMBLYMNG_HPP_
#define TRROM_REDUCEDBASISASSEMBLYMNG_HPP_

#include "TRROM_AssemblyManager.hpp"

namespace trrom
{

class ReducedBasisPDE;
class ReducedBasisData;
class ReducedBasisInterface;
class ReducedBasisObjectiveOperators;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class ReducedBasisAssemblyMng : public trrom::AssemblyManager
{
public:
    ReducedBasisAssemblyMng(const std::shared_ptr<trrom::ReducedBasisData> & data_,
                            const std::shared_ptr<trrom::ReducedBasisInterface> & interface_,
                            const std::shared_ptr<trrom::ReducedBasisObjectiveOperators> & objective_,
                            const std::shared_ptr<trrom::ReducedBasisPDE> & partial_differential_equation_);
    virtual ~ReducedBasisAssemblyMng();

    void updateLowFidelityModel();
    trrom::types::fidelity_t fidelity() const;
    void fidelity(trrom::types::fidelity_t input_);

    int getHessianCounter() const;
    void updateHessianCounter();
    int getGradientCounter() const;
    void updateGradientCounter();
    int getObjectiveCounter() const;
    void updateObjectiveCounter();

    double objective(const std::shared_ptr<trrom::Vector<double> > & control_,
                     const double & tolerance_,
                     bool & inexactness_violated_);
    void gradient(const std::shared_ptr<trrom::Vector<double> > & control_,
                  const std::shared_ptr<trrom::Vector<double> > & gradient_,
                  const double & tolerance_,
                  bool & inexactness_violated_);
    void hessian(const std::shared_ptr<trrom::Vector<double> > & control_,
                 const std::shared_ptr<trrom::Vector<double> > & vector_,
                 const std::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                 const double & tolerance_,
                 bool & inexactness_violated_);


    int getLowFidelitySolveCounter() const;
    void updateLowFidelitySolveCounter();
    int getHighFidelitySolveCounter() const;
    void updateHighFidelitySolveCounter();
    int getLowFidelityAdjointSolveCounter() const;
    void updateLowFidelityAdjointSolveCounter();
    int getHighFidelityAdjointSolveCounter() const;
    void updateHighFidelityAdjointSolveCounter();
    int getLowFidelityJacobianSolveCounter() const;
    void updateLowFidelityJacobianSolveCounter();
    int getHighFidelityJacobianSolveCounter() const;
    void updateHighFidelityJacobianSolveCounter();
    int getLowFidelityAdjointJacobianSolveCounter() const;
    void updateLowFidelityAdjointJacobianSolveCounter();
    int getHighFidelityAdjointJacobianSolveCounter() const;
    void updateHighFidelityAdjointJacobianSolveCounter();

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

    void solveLowFidelityProblem(const trrom::Vector<double> & control_);
    void solveHighFidelityProblem(const trrom::Vector<double> & control_);
    void solveLowFidelityAdjointProblem(const trrom::Vector<double> & control_);
    void solveHighFidelityAdjointProblem(const trrom::Vector<double> & control_);
    void applyInverseJacobianState(const trrom::Vector<double> & control_, const trrom::Vector<double> & rhs_);
    void applyInverseAdjointJacobianState(const trrom::Vector<double> & control_, const trrom::Vector<double> & rhs_);

private:
    bool m_UseFullNewtonHessian;

    int m_HessianCounter;
    int m_GradientCounter;
    int m_ObjectiveCounter;
    int m_LowFidelitySolveCounter;
    int m_HighFidelitySolveCounter;
    int m_LowFidelityAdjointSolveCounter;
    int m_HighFidelityAdjointSolveCounter;
    int m_LowFidelityJacobianSolveCounter;
    int m_HighFidelityJacobianSolveCounter;
    int m_LowFidelityAdjointJacobianSolveCounter;
    int m_HighFidelityAdjointJacobianSolveCounter;

    std::shared_ptr<trrom::Vector<double> > m_Dual;
    std::shared_ptr<trrom::Vector<double> > m_State;
    std::shared_ptr<trrom::Vector<double> > m_DeltaDual;
    std::shared_ptr<trrom::Vector<double> > m_DeltaState;
    std::shared_ptr<trrom::Vector<double> > m_HessWorkVec;
    std::shared_ptr<trrom::Vector<double> > m_StateWorkVec;
    std::shared_ptr<trrom::Vector<double> > m_ControlWorkVec;

    std::shared_ptr<trrom::ReducedBasisPDE> m_PDE;
    std::shared_ptr<trrom::ReducedBasisObjectiveOperators> m_Objective;
    std::shared_ptr<trrom::ReducedBasisInterface> m_ReducedBasisInterface;

private:
    ReducedBasisAssemblyMng(const trrom::ReducedBasisAssemblyMng &);
    trrom::ReducedBasisAssemblyMng & operator=(const trrom::ReducedBasisAssemblyMng &);
};

}

#endif /* TRROM_REDUCEDBASISASSEMBLYMNG_HPP_ */
