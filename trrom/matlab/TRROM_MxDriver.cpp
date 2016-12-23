/*
 * TRROM_MxDriver.cpp
 *
 *  Created on: Dec 20, 2016
 *      Author: maguilo
 */

#include <string>

namespace trrom
{

namespace mx
{

int parseNumberControls(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberControls") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberControls keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberControls"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseNumberStates(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberStates") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberStates keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberStates"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseNumberDuals(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberDuals") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberDuals keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberDuals"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseNumberSlacks(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberSlacks") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberSlacks keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberSlacks"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseMaxNumberSubProblemIterations(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MaxNumberSubProblemIterations") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxNumberSubProblemIterations keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberSubProblemIterations"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseMaxNumberOuterIterations(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MaxNumberOuterIterations") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxNumberOuterIterations keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberOuterIterations"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

double parseMinTrustRegionRadius(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MinTrustRegionRadius") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MinTrustRegionRadius keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MinTrustRegionRadius"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseMaxTrustRegionRadius(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MaxTrustRegionRadius") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxTrustRegionRadius keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxTrustRegionRadius"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseTrustRegionContractionScalar(const mxArray* input_)
{
    if(mxGetField(input_, 0, "TrustRegionContractionScalar") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> TrustRegionContractionScalar keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionContractionScalar"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseTrustRegionExpansionScalar(const mxArray* input_)
{
    if(mxGetField(input_, 0, "TrustRegionExpansionScalar") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> TrustRegionExpansionScalar keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionExpansionScalar"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseActualOverPredictedReductionMidBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ActualOverPredictedReductionMidBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ActualOverPredictedReductionMidBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionMidBound"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseActualOverPredictedReductionLowerBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ActualOverPredictedReductionLowerBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ActualOverPredictedReductionLowerBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionLowerBound"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseActualOverPredictedReductionUpperBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ActualOverPredictedReductionUpperBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ActualOverPredictedReductionUpperBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionUpperBound"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseStepTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "StepTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> StepTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StepTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseGradientTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "GradientTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> GradientTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "GradientTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseObjectiveTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ObjectiveTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ObjectiveTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseStagnationTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "StagnationTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> StagnationTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StagnationTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

void parseControlLowerBound(const mxArray* input_, trrom::MxVector & output_)
{
    if(mxGetField(input_, 0, "ControlLowerBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlLowerBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ControlLowerBound"));
    output_.setMxArray(value);
    mxDestroyArray(value);
}

void parseControlUpperBound(const mxArray* input_, trrom::MxVector & output_)
{
    if(mxGetField(input_, 0, "ControlUpperBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlUpperBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ControlUpperBound"));
    output_.setMxArray(value);
    mxDestroyArray(value);
}

void parseReducedObjectiveFunction(const mxArray* input_, std::tr1::shared_ptr<trrom::MxReducedObjectiveOperators> & output_)
{
    if(mxGetField(input_, 0, "ObjectiveFunction") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ObjectiveFunction keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* mx_objective_function = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveFunction"));
    output_.reset(new trrom::MxReducedObjectiveOperators(mx_objective_function));
    mxDestroyArray(mx_objective_function);
}

void parseReducedBasisPartialDifferentialEquation(const mxArray* input_, std::tr1::shared_ptr<trrom::MxReducedBasisPDE> & output_)
{
    if(mxGetField(input_, 0, "PartialDifferentialEquation") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> PartialDifferentialEquation keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* mx_reduced_basis_pde = mxDuplicateArray(mxGetField(input_, 0, "PartialDifferentialEquation"));
    output_.reset(new trrom::MxReducedBasisPDE(mx_reduced_basis_pde));
    mxDestroyArray(mx_reduced_basis_pde);
}

}

class MxTrustRegionReducedOrderModelAlgorithm
{
public:
    virtual ~MxTrustRegionReducedOrderModelAlgorithm()
    {
    }

    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;
};

class MxTrustRegionReducedOrderModelTypeB : public trrom::MxTrustRegionReducedOrderModelAlgorithm
{
public:
    MxTrustRegionReducedOrderModelTypeB()
    {
    }
    virtual ~MxTrustRegionReducedOrderModelTypeB()
    {
    }

    void initialize(const mxArray* inputs_[],
                    trrom::TrustRegionReducedBasis & algorithm_,
                    trrom::TrustRegionStepMng & step_)
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

    void solve(const mxArray* inputs_[], mxArray* outputs_[])
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

    void output(const trrom::ReducedBasisNewtonDataMng & data_, mxArray* outputs_[])
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

        trrom::MxVector & mx_gradient = dynamic_cast<trrom::MxVector>(*data_.getNewGradient());
        mxSetField(outputs_[0], 0, "Gradient", mx_gradient.array());

        mxArray* norm_gradient = mxCreateDoubleScalar(data_.getNormNewGradient());
        mxSetField(outputs_[0], 0, "NormGradient", norm_gradient);
        mxDestroyArray(norm_gradient);

        mxArray* norm_step = mxCreateDoubleScalar(data_.getNormTrialStep());
        mxSetField(outputs_[0], 0, "NormStep", norm_step);
        mxDestroyArray(norm_step);

        trrom::MxVector & mx_controls = dynamic_cast<trrom::MxVector>(*data_.getNewPrimal());
        mxSetField(outputs_[0], 0, "Controls", mx_controls.array());
    }

private:
    void solveOptimizationProblem(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
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
        std::tr1::shared_ptr<trrom::MxReducedObjectiveOperators> objective;
        trrom::mx::parseReducedObjectiveFunction(inputs_[1], objective);
        std::tr1::shared_ptr<trrom::MxReducedBasisPDE> partial_differential_equation;
        trrom::mx::parseReducedBasisPartialDifferentialEquation(inputs_[1], partial_differential_equation);
        std::tr1::shared_ptr<trrom::ReducedBasisAssemblyMng>
            assembly_manager(new trrom::ReducedBasisAssemblyMng(data_, reduced_basis_interface, objective, partial_differential_equation));

        // Set optimization algorithm data
        std::tr1::shared_ptr<trrom::ReducedHessian> hessian(new trrom::ReducedHessian);
        std::tr1::shared_ptr<trrom::KelleySachsStepMng> step_mng(new trrom::KelleySachsStepMng(data_, hessian));
        std::tr1::shared_ptr<trrom::ReducedBasisNewtonDataMng> data_mng(new trrom::ReducedBasisNewtonDataMng(data_, assembly_manager));
        trrom::TrustRegionReducedBasis algorithm(data_, data_mng, step_mng);
        this->initialize(inputs_, algorithm, *step_mng);

        // Solve optimization problem
        algorithm.getMin();

        // Output data
        this->output(*data_mng, outputs_);
    }

private:
    MxTrustRegionReducedOrderModelTypeB(const trrom::MxTrustRegionReducedOrderModelTypeB &);
    trrom::MxTrustRegionReducedOrderModelTypeB & operator=(const trrom::MxTrustRegionReducedOrderModelTypeB & rhs_);
};

}

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTRUST REGION REDUCED ORDER MODEL ALGORITHM\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 2 && nOutput == 1))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES NO INPUTS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    trrom::MxTrustRegionReducedOrderModelTypeB algorithm;
    algorithm.solve(pInput, pOutput);
}
