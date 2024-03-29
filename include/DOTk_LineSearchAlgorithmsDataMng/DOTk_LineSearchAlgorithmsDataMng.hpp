/*
 * DOTk_LineSearchAlgorithmsDataMng.hpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHALGORITHMSDATAMNG_HPP_
#define DOTK_LINESEARCHALGORITHMSDATAMNG_HPP_

#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_Primal;
class DOTk_AssemblyManager;
class DOTk_FirstOrderOperator;

class DOTk_LineSearchAlgorithmsDataMng : public dotk::DOTk_OptimizationDataMng
{
public:
    explicit DOTk_LineSearchAlgorithmsDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_LineSearchAlgorithmsDataMng();

    size_t getHessianEvaluationCounter() const;
    size_t getObjectiveFuncEvalCounter() const;
    size_t getGradientEvaluationCounter() const;
    size_t getEqualityConstraintEvaluationCounter() const;
    size_t getAdjointInverseJacobianWrtStateCounter() const;

    void setUserDefinedGradient();
    virtual void computeGradient();
    virtual Real evaluateObjective();
    virtual Real evaluateObjective(const std::shared_ptr<dotk::Vector<Real> > & input_);
    virtual void computeGradient(const std::shared_ptr<dotk::Vector<Real> > & input_,
                                 const std::shared_ptr<dotk::Vector<Real> > & gradient_);
    virtual const std::shared_ptr<dotk::DOTk_AssemblyManager> & getRoutinesMng() const;

protected:
    std::shared_ptr<dotk::DOTk_AssemblyManager> m_RoutinesMng;
    std::shared_ptr<dotk::DOTk_FirstOrderOperator> m_FirstOrderOperator;

private:
    // unimplemented
    DOTk_LineSearchAlgorithmsDataMng(const dotk::DOTk_LineSearchAlgorithmsDataMng &);
    dotk::DOTk_LineSearchAlgorithmsDataMng operator=(const dotk::DOTk_LineSearchAlgorithmsDataMng &);
};

}

#endif /* DOTK_LINESEARCHALGORITHMSDATAMNG_HPP_ */
