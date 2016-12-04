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
#include "TRROM_ReducedBasisInterface.hpp"
#include "TRROM_SpectralDecompositionMng.hpp"

namespace trrom
{

ReducedBasisInterface::ReducedBasisInterface(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                                             const std::tr1::shared_ptr<trrom::SolverInterface> & solver_,
                                             const std::tr1::shared_ptr<trrom::SpectralDecompositionMng> & mng_) :
        m_Data(data_),
        m_Solver(solver_),
        m_SpectralDecompositionMng(mng_),
        m_DualBasis(data_->createDualOrthonormalBasisCopy()),
        m_ReducedDualSolution(data_->createReducedDualSolutionCopy()),
        m_ReducedDualLeftHandSide(data_->createReducedDualLeftHandSideCopy()),
        m_ReducedDualRightHandSide(data_->createReducedDualRightHandSideCopy()),
        m_StateBasis(data_->createStateOrthonormalBasisCopy()),
        m_ReducedStateSolution(data_->createReducedStateSolutionCopy()),
        m_ReducedStateLeftHandSide(data_->createReducedStateLeftHandSideCopy()),
        m_ReducedStateRightHandSide(data_->createReducedStateRightHandSideCopy()),
        m_LeftHandSideBasis(data_->createLeftHandSideOrthonormalBasisCopy()),
        m_FullLeftHandSideMatrix(),
        m_LeftHandSideActiveIndices(data_->createLeftHandSideOrthonormalBasisCopy()),
        m_LeftHandSideDeimCoefficients(data_->createLeftHandSideDeimCoefficientsCopy()),
        m_LeftHandSideBasisTimesIndexMatrix(data_->createLeftHandSideOrthonormalBasisCopy()),
        m_LeftHandSideVectorTimesIndexMatrix(data_->createLeftHandSideSnapshotCopy()),
        m_ReducedDualLeftHandSideEnsemble( {}),
        m_ReducedStateLeftHandSideEnsemble( {})
{
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
    m_SpectralDecompositionMng->solveStateSVD();
    m_SpectralDecompositionMng->computeStateOrthonormalBasis(*m_StateBasis);

    int num_state_basis_vectors = m_StateBasis->getNumCols();
    m_ReducedStateSolution = m_Data->createReducedStateSolutionCopy(num_state_basis_vectors);
    m_ReducedStateRightHandSide = m_Data->createReducedStateRightHandSideCopy(num_state_basis_vectors);

    m_SpectralDecompositionMng->solveLeftHandSideSVD();
    m_SpectralDecompositionMng->computeLeftHandSideOrthonormalBasis(*m_LeftHandSideBasis);

    if(m_SpectralDecompositionMng->areDualSnapshotsCollected() == true)
    {
        m_SpectralDecompositionMng->solveDualSVD();
        m_SpectralDecompositionMng->computeDualOrthonormalBasis(*m_DualBasis);

        int num_dual_basis_vectors = m_DualBasis->getNumCols();
        m_ReducedDualSolution = m_Data->createReducedDualSolutionCopy(num_dual_basis_vectors);
        m_ReducedDualRightHandSide = m_Data->createReducedDualRightHandSideCopy(num_dual_basis_vectors);
    }
}

void ReducedBasisInterface::updateReducedStateRightHandSide()
{
    // Compute reduced right hand side vector
    m_StateBasis->gemv(true, 1., m_Data->getRightHandSideSnapshot(), 0., *m_ReducedStateRightHandSide);
}

void ReducedBasisInterface::updateLeftHandSideDeimDataStructures()
{
    // Apply Discrete Empirical Interpolation Method (DEIM) to compute active indices,
    // (i.e. degrees of freedom (dofs))
    m_LeftHandSideActiveIndices = m_LeftHandSideBasis->create();
    this->applyDiscreteEmpiricalInterpolationMethod();

    // Precompute \Phi_{\mathcal{I}} = \mathcal{I}^{T}\Phi
    int num_columns = m_LeftHandSideBasis->getNumCols();
    m_LeftHandSideVectorTimesIndexMatrix = m_Data->createLeftHandSideSnapshotCopy(num_columns);
    m_LeftHandSideBasisTimesIndexMatrix = m_LeftHandSideBasis->create(num_columns, num_columns);
    m_LeftHandSideActiveIndices->gemm(true, false, 1., *m_LeftHandSideBasis, 0., *m_LeftHandSideBasisTimesIndexMatrix);

    // Allocate LHS coefficient vector, \theta(\mathbf{z}), where \theta(\mathbf{z}) =
    // (P^T\Phi)^{-1})(P^{T}\mathbf{k}(\mathbf{z})). Here, P is the matrix of active
    // indices. This solve will be perform during the online calculations
    m_LeftHandSideDeimCoefficients = m_ReducedStateSolution->create(num_columns);
}

void ReducedBasisInterface::updateReducedLeftHandSideEnsembles()
{
    /* Precompute reduced left hand side (LHS) state and dual matrices, \hat{\mathbf{K}}_{i}
     * (\mathbf{z}) = mathbf{W}^{T}\mathbf{K}_{i}(\mathbf{z})\mathbf{V}. mathbf{W} is the
     * left basis, \mathbf{V} is the right basis, and i\in[1,M]. M is the current number of
     * LHS matrices snapshots. Since a Galerkin Reduced Basis approach is used, \mathbf{W}
     * = \mathbf{V} (i.e. left and right reduced basis are the same). */
    int num_dual_snapshots = m_DualBasis->getNumCols();
    m_ReducedDualLeftHandSide = m_Data->createReducedDualLeftHandSideCopy(num_dual_snapshots, num_dual_snapshots);
    int num_state_snapshots = m_StateBasis->getNumCols();
    m_ReducedStateLeftHandSide = m_Data->createReducedStateLeftHandSideCopy(num_state_snapshots, num_state_snapshots);

    int num_dual_degrees_of_freedom = m_DualBasis->getNumRows();
    std::tr1::shared_ptr<trrom::Matrix<double> > dual_work_matrix =
            m_DualBasis->create(num_state_snapshots, num_dual_degrees_of_freedom);

    int num_state_degrees_of_freedom = m_StateBasis->getNumRows();
    std::tr1::shared_ptr<trrom::Matrix<double> > state_work_matrix =
            m_StateBasis->create(num_state_snapshots, num_state_degrees_of_freedom);

    m_ReducedDualLeftHandSideEnsemble.clear();
    m_ReducedStateLeftHandSideEnsemble.clear();
    for(int index = 0; index < num_state_snapshots; ++index)
    {
        // TODO: Transform left hand side snapshot into matrix
        //m_FullLeftHandSideMatrix = m_SpectralDecompositionMng->getLeftHandSideSnapshotEnsemble()->transform(index);

        m_FullLeftHandSideMatrix->gemm(true, false, 1., *m_StateBasis, 0., *state_work_matrix);
        state_work_matrix->gemm(false, false, 1., *m_StateBasis, 0., *m_ReducedStateLeftHandSide);
        m_ReducedStateLeftHandSideEnsemble.push_back(m_ReducedStateLeftHandSide);

        m_FullLeftHandSideMatrix->gemm(true, false, 1., *m_DualBasis, 0., *dual_work_matrix);
        dual_work_matrix->gemm(false, false, 1., *m_StateBasis, 0., *m_ReducedDualLeftHandSide);
        m_ReducedDualLeftHandSideEnsemble.push_back(m_ReducedDualLeftHandSide);

        m_ReducedDualLeftHandSide->fill(0);
        m_ReducedStateLeftHandSide->fill(0);
    }
}

void ReducedBasisInterface::solveLowFidelityPDE(std::tr1::shared_ptr<trrom::Vector<double> > & high_fidelity_solution_)
{
    /*! Solve for \theta(\mathbf{z})\in\mathbb{R}^{M}, where M = number of left hand
     * side (LHS) snapshots. \theta(\mathbf{z}) = \Phi_{\mathcal{I}}^{-1}\mathbf{k}
     * _{\mathcal{I}}(\mathbf{z}). Here, \Phi_{\mathcal{I}} = \mathcal{I}^{T}\Phi and
     * \mathbf{k}_{\mathcal{I}}(\mathbf{z}) = \mathcal{I}^{T}\mathbf{k}(\mathbf{z}).
     * Recall that \mathcal{I} is the matrix of active indices, \mathbf{k}(\mathbf{z})
     * = vec(\mathbf{K}(\mathbf{z})), \mathbf{K}(\mathbf{z}) is the LHS matrix, and
     * \Phi is the set of LHS matrix snapshots in vector format. */
    m_LeftHandSideDeimCoefficients->fill(0);
    m_LeftHandSideActiveIndices->gemv(true, 1., m_Data->getLeftHandSideSnapshot(), 0., *m_LeftHandSideVectorTimesIndexMatrix);
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
    m_StateBasis->gemv(false, 1., *m_ReducedStateSolution, 0., *high_fidelity_solution_);
}

void ReducedBasisInterface::solveLowFidelityAdjointPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & high_fidelity_rhs_,
                                                       std::tr1::shared_ptr<trrom::Vector<double> > & high_fidelity_solution_)
{
    if(m_SpectralDecompositionMng->areDualSnapshotsCollected() == false)
    {
        return;
    }
    // Compute reduced right hand side (RHS) vector of adjoint system of equations.
    m_DualBasis->gemv(true, 1., *high_fidelity_rhs_, 0., *m_ReducedDualRightHandSide);

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
    m_DualBasis->gemv(false, 1., *m_ReducedDualSolution, 0., *high_fidelity_solution_);
}

void ReducedBasisInterface::applyDiscreteEmpiricalInterpolationMethod()
{
    /*! Apply Discrete Empirical Interpolation Method (DEIM) to parameter dependent matrix */
    std::tr1::shared_ptr<trrom::Vector<double> > residual = m_Data->createLeftHandSideSnapshotCopy();
    residual->update(1., m_LeftHandSideBasis->vector(0), 0.);
    residual->modulus();
    int max_index = 0;
    residual->max(max_index);

    int num_basis_vectors = m_LeftHandSideBasis->getNumCols();
    std::tr1::shared_ptr<trrom::Vector<double> > active_indices =
            m_Data->createLeftHandSideSnapshotCopy(num_basis_vectors);
    (*active_indices)[0] = max_index;
    m_LeftHandSideActiveIndices->fill(0);
    m_LeftHandSideActiveIndices->replaceGlobalValue(max_index, 0, 1.);

    int num_degress_of_freedom = m_LeftHandSideBasis->getNumRows();
    std::tr1::shared_ptr<trrom::Vector<double> > reduced_snapshot =
            m_Data->createLeftHandSideSnapshotCopy(num_basis_vectors);

    for(int basis_vector_index = 1; basis_vector_index < num_basis_vectors; ++basis_vector_index)
    {
        std::tr1::shared_ptr<trrom::Matrix<double> > P =
                m_LeftHandSideBasis->create(num_degress_of_freedom, basis_vector_index);
        for(int index = 0; index < basis_vector_index; ++index)
        {
            P->insert(m_LeftHandSideActiveIndices->vector(index));
        }
        // Compute \mathbf{A} = \mathbf{P}(:,1:index-1)^{T}\mathbf{U}
        std::tr1::shared_ptr<trrom::Matrix<double> > A =
                m_LeftHandSideBasis->create(basis_vector_index, num_basis_vectors);
        P->gemm(true, false, 1., *m_LeftHandSideBasis, 0., *A);
        // Compute \vec{rhs} = \mathbf{P}(:,1:basis_vector-1)^{T}\mathbf{U}(:,basis_vector)
        std::tr1::shared_ptr<trrom::Vector<double> > rhs = reduced_snapshot->create(basis_vector_index);
        P->gemv(true, 1., m_LeftHandSideBasis->vector(basis_vector_index), 0., *rhs);
        // Solve \mathbf{A}\vec{lhs}=\vec{rhs}
        reduced_snapshot->fill(0);
        m_Solver->solve(*A, *rhs, *reduced_snapshot);
        // Compute residual, \vec{r} = \mathbf{U}(:,basis_vector) - \mathbf{U}\vec{lhs}
        m_LeftHandSideBasis->gemv(false, -1., *reduced_snapshot, 0., *residual);
        residual->update(1., m_LeftHandSideBasis->vector(basis_vector_index), 1.);
        // Compute DEIM max_index = \max(\lvert \vec{res} \rvert) and set P(max_index,basis_vector) = 1
        residual->modulus();
        residual->max(max_index);
        (*active_indices)[basis_vector_index] = max_index;
        m_LeftHandSideActiveIndices->replaceGlobalValue(max_index, basis_vector_index, 1.);
    }

    m_Data->setLeftHandSideActiveIndices(*active_indices);
}

const std::tr1::shared_ptr<trrom::ReducedBasisData> & ReducedBasisInterface::data() const
{
    return (m_Data);
}

}
