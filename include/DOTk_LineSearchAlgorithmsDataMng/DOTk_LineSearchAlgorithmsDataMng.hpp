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

template<class Type>
class vector;

class DOTk_Primal;
class DOTk_FirstOrderOperator;
class DOTk_SecondOrderOperator;
class DOTk_OptimizationDataMng;
class DOTk_AssemblyManager;

class DOTk_LineSearchAlgorithmsDataMng : public dotk::DOTk_OptimizationDataMng
{
public:
    explicit DOTk_LineSearchAlgorithmsDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_LineSearchAlgorithmsDataMng();

    size_t getHessianEvaluationCounter() const;
    size_t getObjectiveFuncEvalCounter() const;
    size_t getGradientEvaluationCounter() const;
    size_t getEqualityConstraintEvaluationCounter() const;
    size_t getAdjointInverseJacobianWrtStateCounter() const;

    void setUserDefinedGradient();
    virtual void computeGradient();
    virtual Real evaluateObjective();
    virtual Real evaluateObjective(const std::tr1::shared_ptr<dotk::vector<Real> > & input_);
    virtual void computeGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & getRoutinesMng() const;

protected:
    std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> m_RoutinesMng;
    std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> m_FirstOrderOperator;

private:
    // unimplemented
    DOTk_LineSearchAlgorithmsDataMng(const dotk::DOTk_LineSearchAlgorithmsDataMng &);
    dotk::DOTk_LineSearchAlgorithmsDataMng operator=(const dotk::DOTk_LineSearchAlgorithmsDataMng &);
};

}

#endif /* DOTK_LINESEARCHALGORITHMSDATAMNG_HPP_ */
