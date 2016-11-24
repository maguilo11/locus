/*
 * TRROM_MxUtils.hpp
 *
 *  Created on: Nov 22, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXUTILS_HPP_
#define TRROM_MXUTILS_HPP_

#include <vector>

namespace trrom
{

namespace mx
{

std::vector<char> transpose(const bool & input_)
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

}

}

#endif /* TRROM_MXUTILS_HPP_ */
