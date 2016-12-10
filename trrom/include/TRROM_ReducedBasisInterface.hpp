/*
 * TRROM_ReducedBasisInterface.hpp
 *
 *  Created on: Oct 26, 2016
 *      Author: maguilo
 */

#ifndef TRROM_REDUCEDBASISINTERFACE_HPP_
#define TRROM_REDUCEDBASISINTERFACE_HPP_

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
    ReducedBasisInterface(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                          const std::tr1::shared_ptr<trrom::SolverInterface> & solver_,
                          const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & factory_,
                          const std::tr1::shared_ptr<trrom::SpectralDecompositionMng> & mng_);
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

    const std::tr1::shared_ptr<trrom::ReducedBasisData> & data() const;

private:
    void initialize(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_);

private:
    std::tr1::shared_ptr<trrom::ReducedBasisData> m_Data;
    std::tr1::shared_ptr<trrom::SolverInterface> m_Solver;
    std::tr1::shared_ptr<trrom::LinearAlgebraFactory> m_Factory;
    std::tr1::shared_ptr<trrom::SpectralDecompositionMng> m_SpectralDecompositionMng;
    std::tr1::shared_ptr<trrom::DiscreteEmpiricalInterpolation> m_DiscreteEmpiricalInterpolation;

    std::tr1::shared_ptr<trrom::Matrix<double> > m_DualBasis;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedDualSolution;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_ReducedDualLeftHandSide;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedDualRightHandSide;

    std::tr1::shared_ptr<trrom::Matrix<double> > m_StateBasis;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedStateSolution;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_ReducedStateLeftHandSide;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedStateRightHandSide;

    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideBasis;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_FullLeftHandSideMatrix;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideActiveIndices;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideDeimCoefficients;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideBasisTimesIndexMatrix;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideVectorTimesIndexMatrix;

    std::vector<std::tr1::shared_ptr<trrom::Matrix<double> > > m_ReducedDualLeftHandSideEnsemble;
    std::vector<std::tr1::shared_ptr<trrom::Matrix<double> > > m_ReducedStateLeftHandSideEnsemble;

private:
    ReducedBasisInterface(const trrom::ReducedBasisInterface &);
    trrom::ReducedBasisInterface & operator=(const trrom::ReducedBasisInterface &);
};

}

#endif /* TRROM_REDUCEDBASISINTERFACE_HPP_ */
