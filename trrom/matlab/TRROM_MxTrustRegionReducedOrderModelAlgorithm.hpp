/*
 * TRROM_MxTrustRegionReducedOrderModelAlgorithm.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MXTRUSTREGIONREDUCEDORDERMODELALGORITHM_HPP_
#define TRROM_MXTRUSTREGIONREDUCEDORDERMODELALGORITHM_HPP_

#include <mex.h>

namespace trrom
{

class MxTrustRegionReducedOrderModelAlgorithm
{
public:
    //! MxTrustRegionReducedOrderModelAlgorithm destructor
    virtual ~MxTrustRegionReducedOrderModelAlgorithm()
    {
    }

    //! @name Pure virtual function
    //@{
    /*!
     * MEX interface to Trust Region Reduced-Order Model (TRROM) Algorithm.
     * Parameters:
     *    \param In
     *          input_: const array of MEX array pointers with inputs
     *    \param Out
     *          output_: output array of MEX array pointers
     **/
    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;
    //@}
};

}

#endif /* TRROM_MXTRUSTREGIONREDUCEDORDERMODELALGORITHM_HPP_ */
