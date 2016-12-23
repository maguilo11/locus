/*
 * TRROM_MxTrustRegionReducedOrderModelTypeB.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_ReducedHessian.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_ReducedBasisInterface.hpp"
#include "TRROM_TrustRegionReducedBasis.hpp"
#include "TRROM_ReducedBasisAssemblyMng.hpp"
#include "TRROM_SpectralDecompositionMng.hpp"
#include "TRROM_ReducedBasisNewtonDataMng.hpp"


#include "TRROM_MxVector.hpp"
#include "TRROM_MxDirectSolver.hpp"
#include "TRROM_MxBrandLowRankSVD.hpp"
#include "TRROM_MxReducedBasisPDE.hpp"
#include "TRROM_MxParsingUtilities.hpp"
#include "TRROM_MxLinearAlgebraFactory.hpp"
#include "TRROM_MxReducedObjectiveOperators.hpp"
#include "TRROM_MxSingularValueDecomposition.hpp"
#include "TRROM_MxTrustRegionReducedOrderModelTypeB.hpp"

namespace trrom
{

MxTrustRegionReducedOrderModelTypeB::MxTrustRegionReducedOrderModelTypeB()
{
}

MxTrustRegionReducedOrderModelTypeB::~MxTrustRegionReducedOrderModelTypeB()
{
}

void MxTrustRegionReducedOrderModelTypeB::initialize(const mxArray* inputs_[],
                trrom::TrustRegionStepMng & step_,
                trrom::TrustRegionReducedBasis & algorithm_)
{
    // Set core optimization algorithm data
    int integer_value = trrom::mx::parseMaxNumberOuterIterations(inputs_[0]);
    algorithm_.setMaxNumOptimizationItr(integer_value);
    double scalar_value = trrom::mx::parseGradientTolerance(inputs_[0]);
    algorithm_.setGradientTolerance(scalar_value);
    scalar_value = trrom::mx::parseStagnationTolerance(inputs_[0]);
    algorithm_.setStagnationTolerance(scalar_value);
    scalar_value = trrom::mx::parseStepTolerance(inputs_[0]);
    algorithm_.setTrialStepTolerance(scalar_value);
    scalar_value = trrom::mx::parseObjectiveTolerance(inputs_[0]);

    // Set core trust region step manager data
    integer_value = trrom::mx::parseMaxNumberSubProblemIterations(inputs_[0]);
    step_.setMaxNumTrustRegionSubProblemItr(integer_value);
    scalar_value = trrom::mx::parseMinTrustRegionRadius(inputs_[0]);
    step_.setMinTrustRegionRadius(scalar_value);
    scalar_value = trrom::mx::parseMaxTrustRegionRadius(inputs_[0]);
    step_.setMaxTrustRegionRadius(scalar_value);
    scalar_value = trrom::mx::parseTrustRegionContractionScalar(inputs_[0]);
    step_.setTrustRegionContraction(0.5);
    scalar_value = trrom::mx::parseTrustRegionExpansionScalar(inputs_[0]);
    step_.setTrustRegionExpansion(scalar_value);
    scalar_value = trrom::mx::parseActualOverPredictedReductionMidBound(inputs_[0]);
    step_.setActualOverPredictedReductionMidBound(scalar_value);
    scalar_value = trrom::mx::parseActualOverPredictedReductionLowerBound(inputs_[0]);
    step_.setActualOverPredictedReductionLowerBound(scalar_value);
    scalar_value = trrom::mx::parseActualOverPredictedReductionUpperBound(inputs_[0]);
    step_.setActualOverPredictedReductionUpperBound(scalar_value);
}

void MxTrustRegionReducedOrderModelTypeB::solve(const mxArray* inputs_[], mxArray* outputs_[])
{
    // Allocate dual, state, and control MxVectors
    int num_duals = trrom::mx::parseNumberDuals(inputs_[0]);
    int num_states = trrom::mx::parseNumberStates(inputs_[0]);
    int num_controls = trrom::mx::parseNumberControls(inputs_[0]);

    trrom::MxVector duals(num_duals);
    trrom::MxVector states(num_states);
    trrom::MxVector controls(num_controls);
    std::tr1::shared_ptr<trrom::ReducedBasisData> data(new trrom::ReducedBasisData);
    data->allocateDual(duals);
    data->allocateState(states);
    data->allocateControl(controls);

    // Set lower and upper bounds on controls
    trrom::MxVector lower_bounds(num_controls);
    trrom::mx::parseControlLowerBound(inputs_[0], lower_bounds);
    data->setControlLowerBound(lower_bounds);
    trrom::MxVector upper_bounds(num_controls);
    trrom::mx::parseControlUpperBound(inputs_[0], upper_bounds);
    data->setControlUpperBound(upper_bounds);

    // Solve optimization problem
    this->solveOptimizationProblem(data, inputs_, outputs_);
}

void MxTrustRegionReducedOrderModelTypeB::output(const trrom::ReducedBasisNewtonDataMng & data_, mxArray* outputs_[])
{
    // Create memory allocation for output struc
    const char *field_names[6] =
        { "Iterations", "ObjectiveFunction", "Gradient", "NormGradient", "NormStep", "Control" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 6, field_names);

    mxArray* number_iterations = mxCreateDoubleScalar(data_.getIterationCounter());
    mxSetField(outputs_[0], 0, "Iterations", number_iterations);
    mxDestroyArray(number_iterations);

    mxArray* objective_function_value = mxCreateDoubleScalar(data_.getNewObjectiveFunctionValue());
    mxSetField(outputs_[0], 0, "ObjectiveFunction", objective_function_value);
    mxDestroyArray(objective_function_value);

    trrom::MxVector & mx_gradient = dynamic_cast<trrom::MxVector &>(*data_.getNewGradient());
    mxSetField(outputs_[0], 0, "Gradient", mx_gradient.array());

    mxArray* norm_gradient = mxCreateDoubleScalar(data_.getNormNewGradient());
    mxSetField(outputs_[0], 0, "NormGradient", norm_gradient);
    mxDestroyArray(norm_gradient);

    mxArray* norm_step = mxCreateDoubleScalar(data_.getNormTrialStep());
    mxSetField(outputs_[0], 0, "NormStep", norm_step);
    mxDestroyArray(norm_step);

    trrom::MxVector & mx_controls = dynamic_cast<trrom::MxVector &>(*data_.getNewPrimal());
    mxSetField(outputs_[0], 0, "Controls", mx_controls.array());
}

void MxTrustRegionReducedOrderModelTypeB::solveOptimizationProblem(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                                                                   const mxArray* inputs_[],
                                                                   mxArray* outputs_[])
{
    // Set spectral decomposition manager
    std::tr1::shared_ptr<trrom::MxBrandLowRankSVD> low_rank_svd(new trrom::MxBrandLowRankSVD);
    std::tr1::shared_ptr<trrom::MxLinearAlgebraFactory> linear_algebra_factory(new trrom::MxLinearAlgebraFactory);
    std::tr1::shared_ptr<trrom::MxSingularValueDecomposition> full_rank_svd(new trrom::MxSingularValueDecomposition);
    std::tr1::shared_ptr<trrom::SpectralDecompositionMng>
        spectral_decomposition_mng(new trrom::SpectralDecompositionMng(linear_algebra_factory, full_rank_svd, low_rank_svd));

    // Set reduced basis interface: handles low fidelity partial differential equation solves
    std::tr1::shared_ptr<trrom::MxDirectSolver> solver(new trrom::MxDirectSolver);
    std::tr1::shared_ptr<trrom::ReducedBasisInterface>
        reduced_basis_interface(new trrom::ReducedBasisInterface(data_, solver, linear_algebra_factory, spectral_decomposition_mng));

    // Set reduced basis assembly manager: handles objective, gradient, and Hessian evaluations
    std::tr1::shared_ptr<trrom::ReducedBasisObjective> objective;
    trrom::mx::parseReducedObjectiveFunction(inputs_[1], objective);
    std::tr1::shared_ptr<trrom::ReducedBasisPDE> partial_differential_equation;
    trrom::mx::parseReducedBasisPartialDifferentialEquation(inputs_[1], partial_differential_equation);
    std::tr1::shared_ptr<trrom::ReducedBasisAssemblyMng>
        assembly_manager(new trrom::ReducedBasisAssemblyMng(data_, reduced_basis_interface, objective, partial_differential_equation));

    // Set optimization algorithm data
    std::tr1::shared_ptr<trrom::ReducedHessian> hessian(new trrom::ReducedHessian);
    std::tr1::shared_ptr<trrom::KelleySachsStepMng> step_mng(new trrom::KelleySachsStepMng(data_, hessian));
    std::tr1::shared_ptr<trrom::ReducedBasisNewtonDataMng> data_mng(new trrom::ReducedBasisNewtonDataMng(data_, assembly_manager));
    trrom::TrustRegionReducedBasis algorithm(data_, step_mng, data_mng);
    this->initialize(inputs_, *step_mng, algorithm);

    // Solve optimization problem
    algorithm.getMin();

    // Output data
    this->output(*data_mng, outputs_);
}

}
