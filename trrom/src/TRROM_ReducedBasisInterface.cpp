/*ctiv
 * TRROM_ReducedBasisInterface.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: maguilo
 */

#include "TRROM_Matrix.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_SolverInterface.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_LinearAlgebraFactory.hpp"
#include "TRROM_ReducedBasisInterface.hpp"
#include "TRROM_SpectralDecompositionMng.hpp"
#include "TRROM_DiscreteEmpiricalInterpolation.hpp"

namespace trrom
{

ReducedBasisInterface::ReducedBasisInterface(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                                             const std::tr1::shared_ptr<trrom::SolverInterface> & solver_,
                                             const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & factory_,
                                             const std::tr1::shared_ptr<trrom::SpectralDecompositionMng> & mng_) :
        m_Data(data_),
        m_Solver(solver_),
        m_Factory(factory_),
        m_SpectralDecompositionMng(mng_),
        m_DiscreteEmpiricalInterpolation(new trrom::DiscreteEmpiricalInterpolation(solver_, factory_)),
        m_DualBasis(),
        m_ReducedDualSolution(),
        m_ReducedDualLeftHandSide(),
        m_ReducedDualRightHandSide(),
        m_StateBasis(),
        m_ReducedStateSolution(),
        m_ReducedStateLeftHandSide(),
        m_ReducedStateRightHandSide(),
        m_LeftHandSideBasis(),
        m_FullLeftHandSideMatrix(),
        m_LeftHandSideActiveIndices(),
        m_LeftHandSideDeimCoefficients(),
        m_LeftHandSideBasisTimesIndexMatrix(),
        m_LeftHandSideVectorTimesIndexMatrix(),
        m_ReducedDualLeftHandSideEnsemble({}),
        m_ReducedStateLeftHandSideEnsemble({})
{
    this->initialize(data_);
}

ReducedBasisInterface::~ReducedBasisInterface()
{
}

trrom::types::fidelity_t ReducedBasisInterface::fidelity() const
{
    return (m_Data->fidelity());
}

void ReducedBasisInterface::fidelity(trrom::types::fidelity_t input_)
{
    m_Data->fidelity(input_);
}

void ReducedBasisInterface::storeDualSnapshot(const trrom::Vector<double> & dual_)
{
    m_SpectralDecompositionMng->storeDualSnapshot(dual_);
}

void ReducedBasisInterface::storeStateSnapshot(const trrom::Vector<double> & state_)
{
    m_SpectralDecompositionMng->storeStateSnapshot(state_);
}

void ReducedBasisInterface::storeLeftHandSideSnapshot(const trrom::Vector<double> & lhs_)
{
    m_SpectralDecompositionMng->storeLeftHandSideSnapshot(lhs_);
}

void ReducedBasisInterface::updateOrthonormalBases()
{
    m_SpectralDecompositionMng->solveStateSingularValueDecomposition();
    m_SpectralDecompositionMng->computeStateOrthonormalBasis(*m_StateBasis);

    int num_state_basis_vectors = m_StateBasis->getNumCols();
    m_Factory->buildLocalVector(num_state_basis_vectors, m_ReducedStateSolution);
    m_Factory->buildLocalVector(num_state_basis_vectors, m_ReducedStateRightHandSide);

    m_SpectralDecompositionMng->solveLeftHandSideSingularValueDecomposition();
    m_SpectralDecompositionMng->computeLeftHandSideOrthonormalBasis(*m_LeftHandSideBasis);

    if(m_SpectralDecompositionMng->areDualSnapshotsCollected() == true)
    {
        m_SpectralDecompositionMng->solveDualSingularValueDecomposition();
        m_SpectralDecompositionMng->computeDualOrthonormalBasis(*m_DualBasis);

        int num_dual_basis_vectors = m_DualBasis->getNumCols();
        m_Factory->buildLocalVector(num_dual_basis_vectors, m_ReducedDualSolution);
        m_Factory->buildLocalVector(num_dual_basis_vectors, m_ReducedDualRightHandSide);
    }
}

void ReducedBasisInterface::updateReducedStateRightHandSide()
{
    // Compute reduced right hand side vector
    m_StateBasis->gemv(true, 1., *m_Data->getRightHandSideSnapshot(), 0., *m_ReducedStateRightHandSide);
}

void ReducedBasisInterface::updateLeftHandSideDeimDataStructures()
{
    // Apply Discrete Empirical Interpolation Method (DEIM) to compute active indices,
    // (i.e. degrees of freedom (dofs))
    std::tr1::shared_ptr<trrom::Vector<double> > active_indices;
    m_LeftHandSideActiveIndices = m_LeftHandSideBasis->create();
    m_DiscreteEmpiricalInterpolation->apply(m_LeftHandSideBasis, m_LeftHandSideActiveIndices, active_indices);
    m_Data->setLeftHandSideActiveIndices(*active_indices);

    // Precompute \Phi_{\mathcal{I}} = \mathcal{I}^{T}\Phi
    int num_columns = m_LeftHandSideBasis->getNumCols();
    m_Factory->buildLocalMatrix(num_columns, num_columns, m_LeftHandSideBasisTimesIndexMatrix);
    m_LeftHandSideActiveIndices->gemm(true, false, 1., *m_LeftHandSideBasis, 0., *m_LeftHandSideBasisTimesIndexMatrix);

    // Allocate LHS coefficient vector, \theta(\mathbf{z}), where \theta(\mathbf{z}) =
    // (P^T\Phi)^{-1})(P^{T}\mathbf{k}(\mathbf{z})). Here, P is the matrix of active
    // indices. This solve will be perform during the online calculations
    m_Factory->buildLocalVector(num_columns, m_LeftHandSideDeimCoefficients);
    m_Factory->buildLocalVector(num_columns, m_LeftHandSideVectorTimesIndexMatrix);
}

void ReducedBasisInterface::updateReducedLeftHandSideEnsembles()
{
    /*! Precompute reduced left hand side (LHS) state and dual matrices, \hat{\mathbf{K}}_{i}
     * (\mathbf{z}) = mathbf{W}^{T}\mathbf{K}_{i}(\mathbf{z})\mathbf{V}. mathbf{W} is the
     * left basis, \mathbf{V} is the right basis, and i\in[1,M]. M is the current number of
     * LHS matrices snapshots. Since a Galerkin Reduced Basis approach is used, \mathbf{W}
     * = \mathbf{V} (i.e. left and right reduced basis are the same). */
    int num_dual_snapshots = m_DualBasis->getNumCols();
    m_Factory->buildLocalMatrix(num_dual_snapshots, num_dual_snapshots, m_ReducedDualLeftHandSide);
    int num_state_snapshots = m_StateBasis->getNumCols();
    m_Factory->buildLocalMatrix(num_state_snapshots, num_state_snapshots, m_ReducedStateLeftHandSide);

    std::tr1::shared_ptr<trrom::Matrix<double> > dual_work_matrix;
    m_Factory->buildMultiVector(num_dual_snapshots, m_Data->dual(), dual_work_matrix);

    std::tr1::shared_ptr<trrom::Matrix<double> > state_work_matrix;
    m_Factory->buildMultiVector(num_state_snapshots, m_Data->state(), state_work_matrix);

    m_ReducedDualLeftHandSideEnsemble.clear();
    m_ReducedStateLeftHandSideEnsemble.clear();
    int num_state_degrees_of_freedom = m_StateBasis->getNumRows();
    for(int index = 0; index < num_state_snapshots; ++index)
    {
        // Reshapes left hand side snapshot into m-by-n matrix
        m_Factory->reshape(num_state_degrees_of_freedom,
                           num_state_degrees_of_freedom,
                           m_SpectralDecompositionMng->getLeftHandSideSnapshot(index),
                           m_FullLeftHandSideMatrix);

        m_FullLeftHandSideMatrix->gemm(true, false, 1., *m_StateBasis, 0., *state_work_matrix);
        state_work_matrix->gemm(true, false, 1., *m_StateBasis, 0., *m_ReducedStateLeftHandSide);
        m_ReducedStateLeftHandSideEnsemble.push_back(m_ReducedStateLeftHandSide);

        m_FullLeftHandSideMatrix->gemm(true, false, 1., *m_DualBasis, 0., *dual_work_matrix);
        dual_work_matrix->gemm(true, false, 1., *m_StateBasis, 0., *m_ReducedDualLeftHandSide);
        m_ReducedDualLeftHandSideEnsemble.push_back(m_ReducedDualLeftHandSide);

        m_ReducedDualLeftHandSide->fill(0);
        m_ReducedStateLeftHandSide->fill(0);
    }
}

void ReducedBasisInterface::solveLowFidelityProblem(trrom::Vector<double> & high_fidelity_solution_)
{
    /*! Solve for \theta(\mathbf{z})\in\mathbb{R}^{M}, where M = number of left hand
     * side (LHS) snapshots. \theta(\mathbf{z}) = \Phi_{\mathcal{I}}^{-1}\mathbf{k}
     * _{\mathcal{I}}(\mathbf{z}). Here, \Phi_{\mathcal{I}} = \mathcal{I}^{T}\Phi and
     * \mathbf{k}_{\mathcal{I}}(\mathbf{z}) = \mathcal{I}^{T}\mathbf{k}(\mathbf{z}).
     * Recall that \mathcal{I} is the matrix of active indices, \mathbf{k}(\mathbf{z})
     * = vec(\mathbf{K}(\mathbf{z})), \mathbf{K}(\mathbf{z}) is the LHS matrix, and
     * \Phi is the set of LHS matrix snapshots in vector format. */
    m_LeftHandSideActiveIndices->gemv(true, 1., *m_Data->getLeftHandSideSnapshot(), 0., *m_LeftHandSideVectorTimesIndexMatrix);
    m_Solver->solve(*m_LeftHandSideBasisTimesIndexMatrix,
                    *m_LeftHandSideVectorTimesIndexMatrix,
                    *m_LeftHandSideDeimCoefficients);

    // Compute current reduced LHS matrix, \tilde{\mathbf{K}}(\mathbf{z}) = \sum_{i=1}^{M}
    // \theta_{i}(\mathbf{z})(mathbf{W}^{T}\mathbf{K}_{i}(\mathbf{z})\mathbf{V}), where M
    // is the number of LHS snapshots
    m_ReducedStateLeftHandSide->fill(0.);
    int num_deim_coefficients = m_LeftHandSideDeimCoefficients->size();
    for(int index = 0; index < num_deim_coefficients; ++index)
    {
        double this_coefficient = (*m_LeftHandSideDeimCoefficients)[index];
        m_ReducedStateLeftHandSide->update(this_coefficient, *m_ReducedStateLeftHandSideEnsemble[index], 1.);
    }

    // Solve reduced system of state equations, \tilde{\mathbf{K}}(\mathbf{z})\theta_u =
    // \tilde{\mathbf{f}}, where \tilde{\mathbf{f}}=\mathbf{W}^{T}\mathbf{f}
    m_ReducedStateSolution->fill(0);
    m_Solver->solve(*m_ReducedStateLeftHandSide, *m_ReducedStateRightHandSide, *m_ReducedStateSolution);

    // Compute state approximation, \mathbf{u}=\mathbf{V}\theta_u(\mathbf{z})
    m_StateBasis->gemv(false, 1., *m_ReducedStateSolution, 0., high_fidelity_solution_);
}

void ReducedBasisInterface::solveLowFidelityAdjointProblem(const trrom::Vector<double> & high_fidelity_rhs_,
                                                           trrom::Vector<double> & low_fidelity_solution_)
{
    if(m_SpectralDecompositionMng->areDualSnapshotsCollected() == false)
    {
        return;
    }
    // Compute reduced right hand side (RHS) vector of adjoint system of equations.
    m_StateBasis->gemv(true, 1., high_fidelity_rhs_, 0., *m_ReducedDualRightHandSide);

    // Compute current reduced LHS dual matrix, \tilde{\mathbf{K}}^{ast}(\mathbf{z}) =
    // \sum_{i=1}^{M}\theta_{i}(\mathbf{z})(mathbf{W}_{\lambda}^{T}\mathbf{K}_{i}
    // (\mathbf{z}) \mathbf{V}_{\lambda}), where M is the number of LHS snapshots
    m_ReducedDualLeftHandSide->fill(0.);
    int num_deim_coefficients = m_LeftHandSideDeimCoefficients->size();
    for(int index = 0; index < num_deim_coefficients; ++index)
    {
        double this_coefficient = (*m_LeftHandSideDeimCoefficients)[index];
        m_ReducedDualLeftHandSide->update(this_coefficient, *m_ReducedDualLeftHandSideEnsemble[index], 1.);
    }

    // Solve reduced dual system of equations, \tilde{\mathbf{K}}(\mathbf{z})
    // \theta_{\lambda} = \tilde{\mathbf{f}_{\lambda}}, where \tilde{\mathbf{f}
    // _{\lambda}} = \mathbf{W}_{\lambda}^{T}\mathbf{f}_{\lambda}
    m_Solver->solve(*m_ReducedDualLeftHandSide, *m_ReducedDualRightHandSide, *m_ReducedDualSolution);

    // Compute dual approximation, i.e. \lambda = \mathbf{V}_{\lambda}\theta_{\lambda}(\mathbf{z})
    m_DualBasis->gemv(false, 1., *m_ReducedDualSolution, 0., low_fidelity_solution_);
}

void ReducedBasisInterface::applyLowFidelityInverseJacobian(const trrom::Vector<double> & high_fidelity_rhs_,
                                                            trrom::Vector<double> & low_fidelity_solution_)
{
    // Compute reduced right hand side (RHS) vector of adjoint system of equations.
    m_StateBasis->gemv(true, 1., high_fidelity_rhs_, 0., *m_ReducedStateRightHandSide);

    /* Solve reduced system of equations, \tilde{\mathbf{J}}(\mathbf{z})\theta_{du} =
     * \tilde{\mathbf{g_z}}, where \tilde{\mathbf{f}}=\mathbf{W}^{T}\mathbf{g_z} */
    m_ReducedStateSolution->fill(0);
    m_Solver->solve(*m_ReducedStateLeftHandSide, *m_ReducedStateRightHandSide, *m_ReducedStateSolution);

    // Compute state approximation, \mathbf{du}=\mathbf{V}\theta_{du}(\mathbf{z})
    m_StateBasis->gemv(false, 1., *m_ReducedStateSolution, 0., low_fidelity_solution_);
}

void ReducedBasisInterface::applyLowFidelityInverseAdjointJacobian(const trrom::Vector<double> & high_fidelity_rhs_,
                                                                   trrom::Vector<double> & low_fidelity_solution_)
{
    // Compute reduced right hand side (RHS) vector of adjoint system of equations.
    m_StateBasis->gemv(true, 1., high_fidelity_rhs_, 0., *m_ReducedDualRightHandSide);

    /* Solve c_u(u(variables_); variables_) dlambda = -(L_uu(u(variables_), variables_, lambda(variables_)) du
     + L_uz(u(variables_), variables_, lambda(variables_)) trial_step_) for dlambda */
    m_ReducedDualSolution->fill(0);
    m_Solver->solve(*m_ReducedDualLeftHandSide, *m_ReducedDualRightHandSide, *m_ReducedDualSolution);

    // Compute state approximation, \mathbf{du}=\mathbf{V}\theta_{du}(\mathbf{z})
    m_DualBasis->gemv(false, 1., *m_ReducedDualSolution, 0., low_fidelity_solution_);
}

const std::tr1::shared_ptr<trrom::ReducedBasisData> & ReducedBasisInterface::data() const
{
    return (m_Data);
}

void ReducedBasisInterface::initialize(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_)
{
    const int num_vectors = 1;
    m_Factory->buildMultiVector(num_vectors, data_->dual(), m_DualBasis);
    m_Factory->buildMultiVector(num_vectors, data_->state(), m_StateBasis);
    m_Factory->buildMultiVector(num_vectors, data_->getLeftHandSideSnapshot(), m_LeftHandSideBasis);
}
}
