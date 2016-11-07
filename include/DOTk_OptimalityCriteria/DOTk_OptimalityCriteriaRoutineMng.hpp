/*
 * DOTk_OptimalityCriteriaRoutineMng.hpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OPTIMALITYCRITERIAROUTINEMNG_HPP_
#define DOTK_OPTIMALITYCRITERIAROUTINEMNG_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimalityCriteriaDataMng;

template <typename Type>
class DOTk_ObjectiveFunction;
template <typename Type>
class DOTk_EqualityConstraint;
template <typename Type>
class DOTk_InequalityConstraint;

class DOTk_OptimalityCriteriaRoutineMng
{
public:
    DOTk_OptimalityCriteriaRoutineMng(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                      const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                                      const std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_);
    ~DOTk_OptimalityCriteriaRoutineMng();

    void solveEqualityConstraint(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_);
    Real evaluateObjectiveFunction(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_);
    void computeObjectiveFunctionGradient(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_);
    void computeMaxControlRelativeDifference(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_);
    Real computeInequalityConstraintResidual(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_);
    void computeInequalityConstraintGradient(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_);

private:
    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_Objective;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_Equality;
    std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > m_Inequality;

private:
    DOTk_OptimalityCriteriaRoutineMng(const dotk::DOTk_OptimalityCriteriaRoutineMng &);
    dotk::DOTk_OptimalityCriteriaRoutineMng & operator=(const dotk::DOTk_OptimalityCriteriaRoutineMng &);
};

}

#endif /* DOTK_OPTIMALITYCRITERIAROUTINEMNG_HPP_ */
