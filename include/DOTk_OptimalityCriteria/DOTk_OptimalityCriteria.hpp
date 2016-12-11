/*
 * DOTk_OptimalityCriteria.hpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OPTIMALITYCRITERIA_HPP_
#define DOTK_OPTIMALITYCRITERIA_HPP_

#include <sstream>
#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_OptimalityCriteriaDataMng;
class DOTk_OptimalityCriteriaRoutineMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;
template<typename ScalarType>
class DOTk_InequalityConstraint;

class DOTk_OptimalityCriteria
{
public:
    DOTk_OptimalityCriteria(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                            const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                            const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                            const std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_);
    ~DOTk_OptimalityCriteria();

    size_t getNumItrDone() const;
    Real getInequalityDual() const;
    Real getInequalityConstraintResidual() const;
    Real getOptimalObjectiveFunctionValue() const;

    void setMoveLimit(Real value_);
    void setDampingParameter(Real value_);
    void setGradientTolerance(Real value_);
    void setBisectionTolerance(Real value_);
    void setFeasibilityTolerance(Real value_);
    void setMaxNumOptimizationItr(size_t value_);
    void setControlStagnationTolerance(Real value_);
    void setInequalityConstraintDualLowerBound(Real value_);
    void setInequalityConstraintDualUpperBound(Real value_);

    bool printDiagnostics() const;
    void enableDiagnostics();

    void gatherSolution(dotk::Vector<Real> & data_) const;
    void gatherGradient(dotk::Vector<Real> & data_) const;
    void gatherOuputStream(std::ostringstream & output_);

    void getMin();

private:
    void updateControl();
    void updateIterationCount();
    void printCurrentProgress();
    void optimalityCriteriaUpdate();
    bool stoppingCriteriaSatisfied();

private:
    bool m_LastTime;
    size_t m_NumItrDone;
    bool m_PrintDiagnostics;
    Real m_ObjectiveFunctionValue;
    std::ostringstream m_OutputStream;
    std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> m_DataMng;
    std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaRoutineMng> m_RoutineMng;

private:
    DOTk_OptimalityCriteria(const dotk::DOTk_OptimalityCriteria &);
    dotk::DOTk_OptimalityCriteria & operator=(const dotk::DOTk_OptimalityCriteria &);
};

}

#endif /* DOTK_OPTIMALITYCRITERIA_HPP_ */
