/*
 * TRROM_MxTrustRegionReducedOrderModelTypeB.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MXTRUSTREGIONREDUCEDORDERMODELTYPEB_HPP_
#define TRROM_MXTRUSTREGIONREDUCEDORDERMODELTYPEB_HPP_

#include <tr1/memory>

#include "TRROM_MxTrustRegionReducedOrderModelAlgorithm.hpp"

namespace trrom
{

class ReducedBasisData;
class TrustRegionStepMng;
class TrustRegionReducedBasis;
class ReducedBasisNewtonDataMng;

class MxTrustRegionReducedOrderModelTypeB : public trrom::MxTrustRegionReducedOrderModelAlgorithm
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxTrustRegionReducedOrderModelTypeB object
     **/
    MxTrustRegionReducedOrderModelTypeB();
    //! MxTrustRegionReducedOrderModelTypeB destructor.
    virtual ~MxTrustRegionReducedOrderModelTypeB();
    //@}

    //! @name Public functions
    //@{
    /*!
     * Sets core data for Trust Region Reduced Order Model (TRROM) algorithm
     * and trust region step manager.
     * Parameters:
     *    \param In
     *          input_: const array of MEX array pointer
     *    \param Out
     *          step: instance of trust region step manager
     *    \param Out
     *          algorithm_: instance of TRROM algorithm
     **/
    void initialize(const mxArray* inputs_[],
                    trrom::TrustRegionStepMng & step_,
                    trrom::TrustRegionReducedBasis & algorithm_);
    /*!
     * Interface to Trust Region Reduced-Order Model (TRROM) Algorithm.
     * Parameters:
     *    \param In
     *          input_: const array of MEX array pointer
     *    \param Out
     *          outputs_: array of MEX array pointer
     **/
    void solve(const mxArray* inputs_[], mxArray* outputs_[]);
    /*!
     * Sets output data
     * Parameters:
     *    \param In
     *          data_: reference to data manager
     *    \param Out
     *          outputs_: array of MEX array pointer
     **/
    void output(const trrom::ReducedBasisNewtonDataMng & data_, mxArray* outputs_[]);
    //@}

private:
    //! @name Private functions
    //@{
    /*!
     * Solves surrogate-based optimization problem
     * Parameters:
     *    \param In
     *          data_: const shared pointer of class ReducedBasisData
     *    \param In
     *          inputs_: const array of MEX array pointer
     *    \param Out
     *          outputs_: array of MEX array pointer
     */
    void solveOptimizationProblem(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                                  const mxArray* inputs_[],
                                  mxArray* outputs_[]);
    //@}

private:
    MxTrustRegionReducedOrderModelTypeB(const trrom::MxTrustRegionReducedOrderModelTypeB &);
    trrom::MxTrustRegionReducedOrderModelTypeB & operator=(const trrom::MxTrustRegionReducedOrderModelTypeB & rhs_);
};

}

#endif /* TRROM_MXTRUSTREGIONREDUCEDORDERMODELTYPEB_HPP_ */
