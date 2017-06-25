/*
 * TRROM_ReducedBasisInterface.hpp
 *
 *  Created on: Oct 26, 2016
 *      Author: maguilo
 */

#ifndef TRROM_REDUCEDBASISINTERFACE_HPP_
#define TRROM_REDUCEDBASISINTERFACE_HPP_

#include <memory>

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class SolverInterface;
class ReducedBasisData;
class LinearAlgebraFactory;
class SpectralDecompositionMng;
class DiscreteEmpiricalInterpolation;

class ReducedBasisInterface
{
public:
    ReducedBasisInterface(const std::shared_ptr<trrom::ReducedBasisData> & data_,
                          const std::shared_ptr<trrom::SolverInterface> & solver_,
                          const std::shared_ptr<trrom::LinearAlgebraFactory> & factory_,
                          const std::shared_ptr<trrom::SpectralDecompositionMng> & mng_);
    ~ReducedBasisInterface();

    trrom::types::fidelity_t fidelity() const;
    void fidelity(trrom::types::fidelity_t input_);
    void storeDualSnapshot(const trrom::Vector<double> & dual_);
    void storeStateSnapshot(const trrom::Vector<double> & state_);
    void storeLeftHandSideSnapshot(const trrom::Vector<double> & lhs_);

    void updateOrthonormalBases();
    void updateReducedStateRightHandSide();
    void updateReducedLeftHandSideEnsembles();
    void updateLeftHandSideDeimDataStructures();

    void solveLowFidelityProblem(trrom::Vector<double> & high_fidelity_solution_);
    void solveLowFidelityAdjointProblem(const trrom::Vector<double> & high_fidelity_rhs_,
                                        trrom::Vector<double> & low_fidelity_solution_);
    void applyLowFidelityInverseJacobian(const trrom::Vector<double> & high_fidelity_rhs_,
                                         trrom::Vector<double> & low_fidelity_solution_);
    void applyLowFidelityInverseAdjointJacobian(const trrom::Vector<double> & high_fidelity_rhs_,
                                                trrom::Vector<double> & low_fidelity_solution_);

    const std::shared_ptr<trrom::ReducedBasisData> & data() const;

private:
    void initialize(const std::shared_ptr<trrom::ReducedBasisData> & data_);

private:
    std::shared_ptr<trrom::ReducedBasisData> m_Data;
    std::shared_ptr<trrom::SolverInterface> m_Solver;
    std::shared_ptr<trrom::LinearAlgebraFactory> m_Factory;
    std::shared_ptr<trrom::SpectralDecompositionMng> m_SpectralDecompositionMng;
    std::shared_ptr<trrom::DiscreteEmpiricalInterpolation> m_DiscreteEmpiricalInterpolation;

    std::shared_ptr<trrom::Matrix<double> > m_DualBasis;
    std::shared_ptr<trrom::Vector<double> > m_ReducedDualSolution;
    std::shared_ptr<trrom::Matrix<double> > m_ReducedDualLeftHandSide;
    std::shared_ptr<trrom::Vector<double> > m_ReducedDualRightHandSide;

    std::shared_ptr<trrom::Matrix<double> > m_StateBasis;
    std::shared_ptr<trrom::Vector<double> > m_ReducedStateSolution;
    std::shared_ptr<trrom::Matrix<double> > m_ReducedStateLeftHandSide;
    std::shared_ptr<trrom::Vector<double> > m_ReducedStateRightHandSide;

    std::shared_ptr<trrom::Matrix<double> > m_LeftHandSideBasis;
    std::shared_ptr<trrom::Matrix<double> > m_FullLeftHandSideMatrix;
    std::shared_ptr<trrom::Matrix<double> > m_LeftHandSideActiveIndices;
    std::shared_ptr<trrom::Vector<double> > m_LeftHandSideDeimCoefficients;
    std::shared_ptr<trrom::Matrix<double> > m_LeftHandSideBasisTimesIndexMatrix;
    std::shared_ptr<trrom::Vector<double> > m_LeftHandSideVectorTimesIndexMatrix;

    std::vector<std::shared_ptr<trrom::Matrix<double> > > m_ReducedDualLeftHandSideEnsemble;
    std::vector<std::shared_ptr<trrom::Matrix<double> > > m_ReducedStateLeftHandSideEnsemble;

private:
    ReducedBasisInterface(const trrom::ReducedBasisInterface &);
    trrom::ReducedBasisInterface & operator=(const trrom::ReducedBasisInterface &);
};

}

#endif /* TRROM_REDUCEDBASISINTERFACE_HPP_ */
