/*
 * TRROM_MxDriver.cpp
 *
 *  Created on: Dec 20, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>
#include "TRROM_MxTrustRegionReducedOrderModelTypeB.hpp"

//! @name MEX Interface
//@{
/*!
 * Main MEX interface to Trust Region Reduced-Order Model (TRROM) Algorithm
 **/
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
//@}
