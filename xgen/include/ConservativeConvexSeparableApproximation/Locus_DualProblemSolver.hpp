/*
 * Locus_DualProblemSolver.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DUALPROBLEMSOLVER_HPP_
#define LOCUS_DUALPROBLEMSOLVER_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;
template<typename ScalarType, typename OrdinalType>
class ConservativeConvexSeparableAppxDataMng;

template<typename ScalarType, typename OrdinalType = size_t>
class DualProblemSolver
{
public:
    virtual ~DualProblemSolver()
    {
    }

    virtual void solve(locus::MultiVector<ScalarType, OrdinalType> & aDual,
                       locus::MultiVector<ScalarType, OrdinalType> & aTrialControl) = 0;
    virtual void update(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void updateObjectiveCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void updateConstraintCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

} // namespace locus

#endif /* LOCUS_DUALPROBLEMSOLVER_HPP_ */
