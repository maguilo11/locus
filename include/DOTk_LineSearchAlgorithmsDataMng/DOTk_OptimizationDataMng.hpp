/*
 * DOTk_OptimizationDataMng.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OPTIMIZATIONDATAMNG_HPP_
#define DOTK_OPTIMIZATIONDATAMNG_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_AssemblyManager;
class DOTk_SecondOrderOperator;

template<typename Type>
class vector;

class DOTk_OptimizationDataMng
{
public:
    DOTk_OptimizationDataMng();
    explicit DOTk_OptimizationDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_OptimizationDataMng();

    void setNumOptimizationItrDone(size_t input_);
    size_t getNumOptimizationItrDone() const;

    void storeCurrentState();
    void setNormTrialStep(Real input_);
    Real getNormTrialStep() const;
    void setNormNewGradient(Real input_);
    Real getNormNewGradient() const;

    virtual void setNewObjectiveFunctionValue(Real value_);
    virtual void setOldObjectiveFunctionValue(Real value_);
    virtual Real getNewObjectiveFunctionValue() const;
    virtual Real getOldObjectiveFunctionValue() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_Primal> & getPrimalStruc() const;

    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getOldDual() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getNewDual() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getMatrixTimesVector() const;

    virtual void setTrialStep(const dotk::vector<Real> & input_);
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getTrialStep() const;

    virtual void setNewPrimal(const dotk::vector<Real> & input_);
    virtual void setOldPrimal(const dotk::vector<Real> & input_);
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getNewPrimal() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getOldPrimal() const;

    virtual void setNewGradient(const dotk::vector<Real> & input_);
    virtual void setOldGradient(const dotk::vector<Real> & input_);
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getNewGradient() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getOldGradient() const;

    virtual Real evaluateObjective();
    virtual Real evaluateObjective(const std::tr1::shared_ptr<dotk::vector<Real> > & input_);

    virtual void computeGradient();
    virtual void computeGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);

    virtual void applyVectorToHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

    virtual size_t getObjectiveFunctionEvaluationCounter() const;

    virtual const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & getRoutinesMng() const;

private:
    size_t m_NumIterations;

    Real m_NormTrialStep;
    Real m_NormNewGradient;
    Real m_OldObjectiveFunction;
    Real m_NewObjectiveFunction;

    std::tr1::shared_ptr<dotk::vector<Real> > m_OldDual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewDual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_TrialStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OldPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OldGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_MatrixTimesVector;

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    // unimplemented
    DOTk_OptimizationDataMng(const dotk::DOTk_OptimizationDataMng &);
    dotk::DOTk_OptimizationDataMng & operator=(const dotk::DOTk_OptimizationDataMng &);
};

}

#endif /* DOTK_OPTIMIZATIONDATAMNG_HPP_ */
