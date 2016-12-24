/*
 * TRROM_MxParserTest.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxParserUtilities.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    std::string msg("\nTESTING MEX PARSER INTERFACE FOR TRROM ALGORITHM\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput_ == 2 && nOutput_ == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES TWO INPUTS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    // *********** TEST USER DEFINED VALUES ***********

    // **** TEST 1: parseNumberDuals ****
    msg.assign("parseNumberDuals");
    int integer_value = trrom::mx::parseNumberDuals(pInput_[0]);
    bool did_test_pass = integer_value == 10;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 2: parseNumberSlacks ****
    msg.assign("parseNumberSlacks");
    integer_value = trrom::mx::parseNumberSlacks(pInput_[0]);
    did_test_pass = integer_value == 11;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 3: parseNumberStates ****
    msg.assign("parseNumberStates");
    integer_value = trrom::mx::parseNumberStates(pInput_[0]);
    did_test_pass = integer_value == 20;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 4: parseNumberControls ****
    msg.assign("parseNumberControls");
    integer_value = trrom::mx::parseNumberControls(pInput_[0]);
    did_test_pass = integer_value == 15;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 5: parseMaxNumberSubProblemIterations ****
    msg.assign("parseMaxNumberSubProblemIterations");
    integer_value = trrom::mx::parseMaxNumberSubProblemIterations(pInput_[0]);
    did_test_pass = integer_value == 5;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 6: parseMaxNumberOuterIterations ****
    msg.assign("parseMaxNumberOuterIterations");
    integer_value = trrom::mx::parseMaxNumberOuterIterations(pInput_[0]);
    did_test_pass = integer_value == 30;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 7: parseMinTrustRegionRadius ****
    msg.assign("parseMinTrustRegionRadius");
    double scalar_value = trrom::mx::parseMinTrustRegionRadius(pInput_[0]);
    did_test_pass = scalar_value == 1e-1;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 8: parseMaxTrustRegionRadius ****
    msg.assign("parseMaxTrustRegionRadius");
    scalar_value = trrom::mx::parseMaxTrustRegionRadius(pInput_[0]);
    did_test_pass = scalar_value == 1e2;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 9: parseTrustRegionContractionScalar ****
    msg.assign("parseTrustRegionContractionScalar");
    scalar_value = trrom::mx::parseTrustRegionContractionScalar(pInput_[0]);
    did_test_pass = scalar_value == 0.22;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 10: parseTrustRegionExpansionScalar ****
    msg.assign("parseTrustRegionExpansionScalar");
    scalar_value = trrom::mx::parseTrustRegionExpansionScalar(pInput_[0]);
    did_test_pass = scalar_value == 4;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 11: parseActualOverPredictedReductionMidBound ****
    msg.assign("parseActualOverPredictedReductionMidBound");
    scalar_value = trrom::mx::parseActualOverPredictedReductionMidBound(pInput_[0]);
    did_test_pass = scalar_value == 0.3;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 12: parseActualOverPredictedReductionLowerBound ****
    msg.assign("parseActualOverPredictedReductionLowerBound");
    scalar_value = trrom::mx::parseActualOverPredictedReductionLowerBound(pInput_[0]);
    did_test_pass = scalar_value == 0.15;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 13: parseActualOverPredictedReductionUpperBound ****
    msg.assign("parseActualOverPredictedReductionUpperBound");
    scalar_value = trrom::mx::parseActualOverPredictedReductionUpperBound(pInput_[0]);
    did_test_pass = scalar_value == 0.8;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 14: parseStepTolerance ****
    msg.assign("parseStepTolerance");
    scalar_value = trrom::mx::parseStepTolerance(pInput_[0]);
    did_test_pass = scalar_value == 0.14;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 15: parseGradientTolerance ****
    msg.assign("parseGradientTolerance");
    scalar_value = trrom::mx::parseGradientTolerance(pInput_[0]);
    did_test_pass = scalar_value == 0.13;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 16: parseObjectiveTolerance ****
    msg.assign("parseObjectiveTolerance");
    scalar_value = trrom::mx::parseObjectiveTolerance(pInput_[0]);
    did_test_pass = scalar_value == 0.12;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 17: parseStagnationTolerance ****
    msg.assign("parseStagnationTolerance");
    scalar_value = trrom::mx::parseStagnationTolerance(pInput_[0]);
    did_test_pass = scalar_value == 0.11;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 18: parseControlLowerBound ****
    msg.assign("parseControlLowerBound");
    mxArray* mx_lower_bound = trrom::mx::parseControlLowerBound(pInput_[0]);
    trrom::MxVector lower_bounds(mx_lower_bound);
    mxDestroyArray(mx_lower_bound);
    const int num_variables = 10;
    trrom::MxVector gold(num_variables, 0.1);
    did_test_pass = trrom::mx::checkResults(gold, lower_bounds);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 19: parseControlUpperBound ****
    msg.assign("parseControlUpperBound");
    mxArray* mx_upper_bound = trrom::mx::parseControlUpperBound(pInput_[0]);
    trrom::MxVector upper_bounds(mx_upper_bound);
    mxDestroyArray(mx_upper_bound);
    gold.fill(1.);
    did_test_pass = trrom::mx::checkResults(gold, upper_bounds);
    trrom::mx::assert_test(msg, did_test_pass);

    // *********** TEST DEFAULT VALUES ***********

    // **** TEST 1d: parseMaxNumberSubProblemIterations ****
    msg.assign("parseMaxNumberSubProblemIterations_default");
    integer_value = trrom::mx::parseMaxNumberSubProblemIterations(pInput_[1]);
    did_test_pass = integer_value == 10;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 2d: parseMaxNumberOuterIterations ****
    msg.assign("parseMaxNumberOuterIterations_default");
    integer_value = trrom::mx::parseMaxNumberOuterIterations(pInput_[1]);
    did_test_pass = integer_value == 50;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 3d: parseMinTrustRegionRadius ****
    msg.assign("parseMinTrustRegionRadius_default");
    scalar_value = trrom::mx::parseMinTrustRegionRadius(pInput_[1]);
    did_test_pass = scalar_value == 1e-4;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 4d: parseMaxTrustRegionRadius ****
    msg.assign("parseMaxTrustRegionRadius_default");
    scalar_value = trrom::mx::parseMaxTrustRegionRadius(pInput_[1]);
    did_test_pass = scalar_value == 1e4;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 5d: parseTrustRegionContractionScalar ****
    msg.assign("parseTrustRegionContractionScalar_default");
    scalar_value = trrom::mx::parseTrustRegionContractionScalar(pInput_[1]);
    did_test_pass = scalar_value == 0.5;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 6d: parseTrustRegionExpansionScalar ****
    msg.assign("parseTrustRegionExpansionScalar_default");
    scalar_value = trrom::mx::parseTrustRegionExpansionScalar(pInput_[1]);
    did_test_pass = scalar_value == 2;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 7d: parseActualOverPredictedReductionMidBound ****
    msg.assign("parseActualOverPredictedReductionMidBound_default");
    scalar_value = trrom::mx::parseActualOverPredictedReductionMidBound(pInput_[1]);
    did_test_pass = scalar_value == 0.25;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 8d: parseActualOverPredictedReductionLowerBound ****
    msg.assign("parseActualOverPredictedReductionLowerBound_default");
    scalar_value = trrom::mx::parseActualOverPredictedReductionLowerBound(pInput_[1]);
    did_test_pass = scalar_value == 0.1;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 9d: parseActualOverPredictedReductionUpperBound ****
    msg.assign("parseActualOverPredictedReductionUpperBound_default");
    scalar_value = trrom::mx::parseActualOverPredictedReductionUpperBound(pInput_[1]);
    did_test_pass = scalar_value == 0.75;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 10d: parseStepTolerance ****
    msg.assign("parseStepTolerance_default");
    scalar_value = trrom::mx::parseStepTolerance(pInput_[1]);
    did_test_pass = scalar_value == 1e-10;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 11d: parseGradientTolerance ****
    msg.assign("parseGradientTolerance_default");
    scalar_value = trrom::mx::parseGradientTolerance(pInput_[1]);
    did_test_pass = scalar_value == 1e-10;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 12d: parseObjectiveTolerance ****
    msg.assign("parseObjectiveTolerance_default");
    scalar_value = trrom::mx::parseObjectiveTolerance(pInput_[1]);
    did_test_pass = scalar_value == 1e-10;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 13d: parseStagnationTolerance ****
    msg.assign("parseStagnationTolerance_default");
    scalar_value = trrom::mx::parseStagnationTolerance(pInput_[1]);
    did_test_pass = scalar_value == 1e-12;
    trrom::mx::assert_test(msg, did_test_pass);
}
