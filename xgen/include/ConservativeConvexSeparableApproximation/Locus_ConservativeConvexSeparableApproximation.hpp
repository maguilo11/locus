/*
 * Locus_ConservativeConvexSeparableApproximation.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CONSERVATIVECONVEXSEPARABLEAPPROXIMATION_HPP_
#define LOCUS_CONSERVATIVECONVEXSEPARABLEAPPROXIMATION_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class PrimalProblemStageMng;
template<typename ScalarType, typename OrdinalType>
class ConservativeConvexSeparableAppxDataMng;

struct ccsa
{
    enum stop_t
    {
        STATIONARITY_TOLERANCE = 1,
        KKT_CONDITIONS_TOLERANCE = 2,
        CONTROL_STAGNATION = 3,
        OBJECTIVE_STAGNATION = 4,
        MAX_NUMBER_ITERATIONS = 5,
        OPTIMALITY_AND_FEASIBILITY_MET = 6,
        NOT_CONVERGED = 7,
    };
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableApproximation
{
public:
    virtual ~ConservativeConvexSeparableApproximation()
    {
    }

    virtual void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
                       locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

} // namespace locus

#endif /* LOCUS_CONSERVATIVECONVEXSEPARABLEAPPROXIMATION_HPP_ */
