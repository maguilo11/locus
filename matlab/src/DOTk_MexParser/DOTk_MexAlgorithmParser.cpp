/*
 * DOTk_MexAlgorithmParser.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"

namespace dotk
{

namespace mex
{

void parseThreadCount(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ThreadCount")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: ThreadCount is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr thread_count;
    thread_count.reset(mxDuplicateArray(mxGetField(options_, 0, "ThreadCount")));
    output_ = static_cast<size_t>(mxGetScalar(thread_count.get()));
    thread_count.release();
}

void parseNumberDuals(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "NumberDuals")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: NumberDuals is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr num_duals;
    num_duals.reset(mxDuplicateArray(mxGetField(options_, 0, "NumberDuals")));
    output_ = static_cast<size_t>(mxGetScalar(num_duals.get()));
    num_duals.release();
}

void parseNumberStates(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "NumberStates")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: NumberStates is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr num_states;
    num_states.reset(mxDuplicateArray(mxGetField(options_, 0, "NumberStates")));
    output_ = static_cast<size_t>(mxGetScalar(num_states.get()));
    num_states.release();
}

void parseMaxNumUpdates(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumberUpdates")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: MaxNumberUpdates is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr max_number_updates;
    max_number_updates.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumberUpdates")));
    output_ = static_cast<size_t>(mxGetScalar(max_number_updates.get()));
    max_number_updates.release();
}

void parseNumberControls(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "NumberControls")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: NumberControls is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr num_controls;
    num_controls.reset(mxDuplicateArray(mxGetField(options_, 0, "NumberControls")));
    output_ = static_cast<size_t>(mxGetScalar(num_controls.get()));
    num_controls.release();
}

void parseMaxNumFeasibleItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumFeasibleItr")) == true)
    {
        output_ = 5;
        std::string msg(" DOTk/MEX WARNING: MaxNumFeasibleItr is NOT Defined. Default = 5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr iterations;
    iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumFeasibleItr")));
    output_ = static_cast<size_t>(mxGetScalar(iterations.get()));
    iterations.release();
}

void parseMaxNumAlgorithmItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumAlgorithmItr")) == true)
    {
        output_ = 100;
        std::string msg(" DOTk/MEX WARNING: MaxNumAlgorithmItr is NOT Defined. Default = 100. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr iterations;
    iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumAlgorithmItr")));
    output_ = static_cast<size_t>(mxGetScalar(iterations.get()));
    iterations.release();
}

void parseMaxNumLineSearchItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumLineSearchItr")) == true)
    {
        output_ = 10;
        std::string msg(" DOTk/MEX WARNING: MaxNumLineSearchItr is NOT Defined. Default = 10. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr iterations;
    iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumLineSearchItr")));
    output_ = static_cast<size_t>(mxGetScalar(iterations.get()));
    iterations.release();
}

void parseMaxNumTrustRegionSubProblemItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumTrustRegionSubProblemItr")) == true)
    {
        output_ = 10;
        std::string msg(" DOTk/MEX WARNING: MaxNumTrustRegionSubProblemItr is NOT Defined. Default = 10. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr iterations;
    iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumTrustRegionSubProblemItr")));
    output_ = static_cast<size_t>(mxGetScalar(iterations.get()));
    iterations.release();
}

void parseFiniteDifferenceDiagnosticsUpperSuperScripts(const mxArray* options_, int & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "FiniteDifferenceDiagnosticsUpperSuperScripts")) == true)
    {
        output_ = 5;
        std::string msg(" DOTk/MEX WARNING: FiniteDifferenceDiagnosticsUpperSuperScripts is NOT Defined. Default = 5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "FiniteDifferenceDiagnosticsUpperSuperScripts")));
    output_ = static_cast<int>(mxGetScalar(factor.get()));
    factor.release();
}

void parseFiniteDifferenceDiagnosticsLowerSuperScripts(const mxArray* options_, int & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "FiniteDifferenceDiagnosticsLowerSuperScripts")) == true)
    {
        output_ = -3;
        std::string msg(" DOTk/MEX WARNING: FiniteDifferenceDiagnosticsLowerSuperScripts is NOT Defined. Default = -3. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "FiniteDifferenceDiagnosticsLowerSuperScripts")));
    output_ = static_cast<int>(mxGetScalar(factor.get()));
    factor.release();
}

void parseGradientTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "GradientTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: GradientTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "GradientTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseResidualTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ResidualTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: ResidualTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "ResidualTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseOptimalityTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "OptimalityTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: OptimalityTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "OptimalityTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseObjectiveTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ObjectiveTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: ObjectiveTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "ObjectiveTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseTrialStepTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TrialStepTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: TrialStepTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "TrialStepTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseFeasibilityTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "FeasibilityTolerance")) == true)
    {
        output_ = 1e-4;
        std::string msg(" DOTk/MEX ERROR: FeasibilityTolerance is NOT Defined. Default = 1e-4. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "FeasibilityTolerance")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseActualReductionTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ActualReductionTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX ERROR: ActualReductionTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "ActualReductionTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseControlStagnationTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ControlStagnationTolerance")) == true)
    {
        output_ = 1e-2;
        std::string msg(" DOTk/MEX ERROR: ControlStagnationTolerance is NOT Defined. Default = 1e-2. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "ControlStagnationTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseMaxTrustRegionRadius(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxTrustRegionRadius")) == true)
    {
        output_ = 1e4;
        std::string msg(" DOTk/MEX WARNING: MaxTrustRegionRadius is NOT Defined. Default = 1e4. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxTrustRegionRadius")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseMinTrustRegionRadius(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MinTrustRegionRadius")) == true)
    {
        output_ = 1e-6;
        std::string msg(" DOTk/MEX WARNING: MinTrustRegionRadius is NOT Defined. Default = 1e-6. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "MinTrustRegionRadius")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseBoundConstraintStepSize(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "BoundConstraintStepSize")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: BoundConstraintStepSize is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "BoundConstraintStepSize")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseInitialTrustRegionRadius(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "InitialTrustRegionRadius")) == true)
    {
        output_ = 1e3;
        std::string msg(" DOTk/MEX WARNING: InitialTrustRegionRadius is NOT Defined. Default = 1e3. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "InitialTrustRegionRadius")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseTrustRegionExpansionFactor(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TrustRegionExpansionFactor")) == true)
    {
        output_ = 2.;
        std::string msg(" DOTk/MEX WARNING: TrustRegionExpansionFactor is NOT Defined. Default = 2. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "TrustRegionExpansionFactor")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseGoldsteinLineSearchConstant(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "GoldsteinLineSearchConstant")) == true)
    {
        output_ = 0.9;
        std::string msg(" DOTk/MEX WARNING: GoldsteinLineSearchConstant is NOT Defined. Default = 0.9. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "GoldsteinLineSearchConstant")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseLineSearchContractionFactor(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "LineSearchContractionFactor")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: LineSearchContractionFactor is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "LineSearchContractionFactor")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseTrustRegionContractionFactor(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TrustRegionContractionFactor")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: TrustRegionContractionFactor is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "TrustRegionContractionFactor")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseLineSearchStagnationTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "LineSearchStagnationTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: LineSearchStagnationTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "LineSearchStagnationTolerance")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseBoundConstraintContractionFactor(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "BoundConstraintContractionFactor")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: BoundConstraintContractionFactor is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "BoundConstraintContractionFactor")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseMinActualOverPredictedReductionRatio(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MinActualOverPredictedReductionRatio")) == true)
    {
        output_ = 0.1;
        std::string msg(" DOTk/MEX WARNING: MinActualOverPredictedReductionRatio is NOT Defined. Default = 0.1. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "MinActualOverPredictedReductionRatio")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseMidActualOverPredictedReductionRatio(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MidActualOverPredictedReductionRatio")) == true)
    {
        output_ = 0.25;
        std::string msg(" DOTk/MEX WARNING: MidActualOverPredictedReductionRatio is NOT Defined. Default = 0.25. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "MidActualOverPredictedReductionRatio")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseMaxActualOverPredictedReductionRatio(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxActualOverPredictedReductionRatio")) == true)
    {
        output_ = 0.75;
        std::string msg(" DOTk/MEX WARNING: MaxActualOverPredictedReductionRatio is NOT Defined. Default = 0.75. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxActualOverPredictedReductionRatio")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseSetInitialTrustRegionRadiusToNormGradFlag(const mxArray* options_, bool & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "SetInitialTrustRegionRadiusToNormGrad")) == true)
    {
        std::string msg(" DOTk/MEX WARNING: SetInitialTrustRegionRadiusToNormGrad is NOT Defined. Default = true. \n");
        mexWarnMsgTxt(msg.c_str());
        output_ = true;
        return;
    }

    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "SetInitialTrustRegionRadiusToNormGrad")));

    std::string flag(mxArrayToString(ptr.get()));
    if(flag.compare("false") == 0)
    {
        output_ = false;
    }
    else if(flag.compare("true") == 0)
    {
        output_ = true;
    }
    else
    {
        std::string msg(" DOTk/MEX WARNING: SetInitialTrustRegionRadiusToNormGrad is NOT Defined. Options are true or false. Default = true. \n");
        mexWarnMsgTxt(msg.c_str());
        output_ = true;
    }

    ptr.release();
}

void parseDualData(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "Dual")) == true)
    {
        output_.fill(0);
        std::string msg(" DOTk/MEX WARNING: Initial Dual Data was NOT Defined. Elements in Dual Container set to zero. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "Dual")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseStateData(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "State")) == true)
    {
        output_.fill(0);
        std::string msg(" DOTk/MEX WARNING: Initial State Data was NOT Defined. Elements in State Container set to zero. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "State")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseControlData(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "Control")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: Initial Control Data was NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "Control")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseObjectiveFunction(const mxArray* options_, dotk::DOTk_MexArrayPtr & ptr_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ObjectiveFunction")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: Objective Function Operators are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    ptr_.reset(mxDuplicateArray(mxGetField(options_, 0, "ObjectiveFunction")));
}

void parseEqualityConstraint(const mxArray* options_, dotk::DOTk_MexArrayPtr & ptr_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "EqualityConstraint")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: Equality Constraint Operators are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    ptr_.reset(mxDuplicateArray(mxGetField(options_, 0, "EqualityConstraint")));
}

void parseInequalityConstraint(const mxArray* options_, dotk::DOTk_MexArrayPtr & ptr_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "InequalityConstraint")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: Inequality Constraint Operators are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    ptr_.reset(mxDuplicateArray(mxGetField(options_, 0, "InequalityConstraint")));
}

void parseProblemType(const mxArray* options_, dotk::types::problem_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ProblemType")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: ProblemType is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "ProblemType")));
    output_ = dotk::mex::getProblemType(type);
    type.release();
}

void parseContainerType(const mxArray* options_, dotk::types::container_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ContainerType")) == true)
    {
        output_ = dotk::types::SERIAL_ARRAY;
        std::string msg(" DOTk/MEX WARNING: ContainerType is NOT Defined. Default = Serial C Array. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "ContainerType")));
    output_ = dotk::mex::getContainerType(type);
    type.release();
}

void parseLineSearchMethod(const mxArray* options_, dotk::types::line_search_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "LineSearchMethod")) == true)
    {
        output_ = dotk::types::BACKTRACKING_CUBIC_INTRP;
        std::string msg(" DOTk/MEX WARNING: LineSearchMethod is NOT Defined. Default = BACKTRACKING CUBIC INTERPOLATION. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "LineSearchMethod")));
    output_ = dotk::mex::getLineSearchMethod(type);
    type.release();
}

void parseTrustRegionMethod(const mxArray* options_, dotk::types::trustregion_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TrustRegionMethod")) == true)
    {
        output_ = dotk::types::TRUST_REGION_DOGLEG;
        std::string msg(" DOTk/MEX WARNING: TrustRegionMethod is NOT Defined. Default = DOGLEG. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "TrustRegionMethod")));
    output_ = dotk::mex::getTrustRegionMethod(type);
    type.release();
}

void parseNonlinearCgMethod(const mxArray* options_, dotk::types::nonlinearcg_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "NonlinearCgMethod")) == true)
    {
        output_ = dotk::types::HAGER_ZHANG_NLCG;
        std::string msg(" DOTk/MEX WARNING: NonlinearCgMethod is NOT Defined. Default = HAGER-ZHANG. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "NonlinearCgMethod")));
    output_ = dotk::mex::getNonlinearCgMethod(type);
    type.release();
}

void parseHessianComputationMethod(const mxArray* options_, dotk::types::hessian_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "HessianComputationMethod")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: HessianComputationMethod is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "HessianComputationMethod")));
    output_ = dotk::mex::getHessianComputationMethod(type);
    type.release();
}

void parseGradientComputationMethod(const mxArray* options_, dotk::types::gradient_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "GradientComputationMethod")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: GradientComputationMethod is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "GradientComputationMethod")));
    output_ = dotk::mex::getGradientComputationMethod(type);
    type.release();
}

void parseBoundConstraintMethod(const mxArray* options_, dotk::types::constraint_method_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "BoundConstraintMethod")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: BoundConstraintMethod is NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr method;
    method.reset(mxDuplicateArray(mxGetField(options_, 0, "BoundConstraintMethod")));
    output_ = dotk::mex::getBoundConstraintMethod(method);
    method.release();
}

void parseDualLowerBound(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualLowerBounds")) == true)
    {
        std::string msg(" DOTk/MEX WARNING: DualLowerBounds are NOT Defined. Default values used. See Users' Manual. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "DualLowerBounds")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseDualUpperBound(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualUpperBounds")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: DualUpperBounds are NOT Defined. Default values used. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "DualUpperBounds")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseStateLowerBound(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "StateLowerBounds")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: StateLowerBounds are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "StateLowerBounds")));
    dotk::mex::setDOTkData(data, output_ );
    data.release();
}

void parseStateUpperBound(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "StateUpperBounds")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: StateUpperBounds are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "StateUpperBounds")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseControlLowerBound(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ControlLowerBounds")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: ControlLowerBounds are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "ControlLowerBounds")));
    dotk::mex::setDOTkData(data, output_ );
    data.release();
}

void parseControlUpperBound(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ControlUpperBounds")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: ControlUpperBounds are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "ControlUpperBounds")));
    dotk::mex::setDOTkData(data, output_);
    data.release();
}

void parseFiniteDifferencePerturbation(const mxArray* options_, dotk::Vector<double> & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "FiniteDifferencePerturbations")) == true)
    {
        std::string msg(" DOTk/MEX ERROR: FiniteDifferencePerturbations are NOT Defined. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "FiniteDifferencePerturbations")));

    for(size_t index = 0; index < output_.size(); ++index)
    {
        output_[index] = mxGetPr(data.get())[index];
    }

    data.release();
}

}

}
