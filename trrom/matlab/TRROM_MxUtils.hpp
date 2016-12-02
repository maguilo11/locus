/*
 * TRROM_MxUtils.hpp
 *
 *  Created on: Nov 22, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXUTILS_HPP_
#define TRROM_MXUTILS_HPP_

#include <mex.h>
#include <vector>
#include <string>
#include <sstream>

namespace trrom
{

namespace mx
{

inline std::vector<char> transpose(const bool & input_)
{
    std::vector<char> output;
    if(input_ == true)
    {
        output.push_back('T');
    }
    else
    {
        output.push_back('N');
    }
    return (output);
}

//! Handles a Matlab exception.
inline void handleException(mxArray* error_, std::string msg_)
{
    if(error_)
    {
        // In the cass of an exception, grab the report
        mxArray* input[1] = {error_};
        mxArray* output[1];
        mexCallMATLABWithObject(1, output, 1, input, "getReport");
        // Turn the report into a string
        mwSize char_limit = 256;
        char report_[char_limit];
        mxGetString(output[0], report_, char_limit);

        /*
           The report has extra information that we don't want.  Hence,
           we eliminate both the first line as well as the last two lines.
           The first line is supposed to say what function this occured in,
           but Matlab gets a little confused since we're doing mex trickery.
           The last two lines will automatically be repeated by mexErrMsgTxt.
        */

        std::string report = report_;
        size_t pos = report.find("\n");
        report = report.substr(pos + 1);
        pos = report.rfind("\n");
        report = report.substr(0, pos);
        pos = report.rfind("\n");
        report = report.substr(0, pos);
        pos = report.rfind("\n");
        report = report.substr(0, pos);
        // Now, tack on our additional error message and then return control to Matlab.
        std::stringstream ss;
        ss << msg_ << std::endl << std::endl << report;
        mexErrMsgTxt(ss.str().c_str());
    }
}

inline void setMxArray(const mxArray* input_, mxArray* output_)
{
    size_t number_of_elements = mxGetNumberOfElements(input_);
    assert(number_of_elements == mxGetNumberOfElements(output_));

    double* input_data = mxGetPr(input_);
    double* output_data = mxGetPr(output_);
    for(size_t index = 0; index < number_of_elements; ++index)
    {
        output_data[index] = input_data[index];
    }
}

}

}

#endif /* TRROM_MXUTILS_HPP_ */
