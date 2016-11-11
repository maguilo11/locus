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
class ReducedBasisObjective;
class ReducedBasisInterface;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class ReducedBasisAssemblyMng : public trrom::AssemblyManager
{
public:
    ReducedBasisAssemblyMng(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                            const std::tr1::shared_ptr<trrom::ReducedBasisInterface> & interface_,
                            const std::tr1::shared_ptr<trrom::ReducedBasisPDE> & pde_,
                            const std::tr1::shared_ptr<trrom::ReducedBasisObjective> & objective_);
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

    double objective(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                     const double & tolerance_,
                     bool & inexactness_violated_);
    void gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                  const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_,
                  const double & tolerance_,
                  bool & inexactness_violated_);
    void hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                 const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                 const std::tr1::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
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

    void useGaussNewtonHessian();

private:
    void computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                  const trrom::Vector<double> & vector_,
                                  trrom::Vector<double> & hess_times_vec_);
    void computeGaussNewtonHessian(const trrom::Vector<double> & control_,
                                   const trrom::Vector<double> & vector_,
                                   trrom::Vector<double> & hess_times_vec_);

    void solveHighFidelityPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_);
    void solveLowFidelityPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_);
    void solveHighFidelityAdjointPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_);
    void solveLowFidelityAdjointPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_);

private:
    bool m_UseFullNewtonHessian;

    int m_HessianCounter;
    int m_GradientCounter;
    int m_ObjectiveCounter;
    int m_LowFidelitySolveCounter;
    int m_HighFidelitySolveCounter;
    int m_LowFidelityAdjointSolveCounter;
    int m_HighFidelityAdjointSolveCounter;

    trrom::types::fidelity_t m_Fidelity;

    std::tr1::shared_ptr<trrom::Vector<double> > m_Dual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_State;
    std::tr1::shared_ptr<trrom::Vector<double> > m_StateWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ControlWorkVec;

    std::tr1::shared_ptr<trrom::Vector<double> > m_RightHandSide;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideSnapshot;

    std::tr1::shared_ptr<trrom::ReducedBasisPDE> m_PDE;
    std::tr1::shared_ptr<trrom::ReducedBasisData> m_Data;
    std::tr1::shared_ptr<trrom::ReducedBasisObjective> m_Objective;
    std::tr1::shared_ptr<trrom::ReducedBasisInterface> m_ReducedBasisInterface;

private:
    ReducedBasisAssemblyMng(const trrom::ReducedBasisAssemblyMng &);
    trrom::ReducedBasisAssemblyMng & operator=(const trrom::ReducedBasisAssemblyMng &);
};

}

#endif /* TRROM_REDUCEDBASISASSEMBLYMNG_HPP_ */
