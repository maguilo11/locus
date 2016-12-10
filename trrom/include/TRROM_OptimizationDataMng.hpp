/*
 * TRROM_OptimizationDataMng.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_OPTIMIZATIONDATAMNG_HPP_
#define TRROM_OPTIMIZATIONDATAMNG_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

class Data;

template<typename ScalarType>
class Vector;

class OptimizationDataMng
{
public:
    OptimizationDataMng();
    explicit OptimizationDataMng(const std::tr1::shared_ptr<trrom::Data> & data_);
    virtual ~OptimizationDataMng();

    int getIterationCounter() const;
    void setIterationCounter(int input_);

    void setNormTrialStep(double input_);
    double getNormTrialStep() const;
    void setNormNewGradient(double input_);
    double getNormNewGradient() const;
    void setStagnationMeasure(double input_);
    double getStagnationMeasure() const;

    void setGradientInexactnessFlag(bool input_);
    bool isGradientInexactnessToleranceExceeded();
    void setGradientInexactnessTolerance(double input);
    double getGradientInexactnessTolerance() const;

    void setObjectiveInexactnessFlag(bool input_);
    bool isObjectiveInexactnessToleranceExceeded();
    void setObjectiveInexactnessTolerance(double input);
    double getObjectiveInexactnessTolerance() const;

    void setNewObjectiveFunctionValue(double value_);
    void setOldObjectiveFunctionValue(double value_);
    double getNewObjectiveFunctionValue() const;
    double getOldObjectiveFunctionValue() const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & getMatrixTimesVector() const;

    void setTrialStep(const trrom::Vector<double> & input_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & getTrialStep() const;

    void setNewPrimal(const trrom::Vector<double> & input_);
    void setOldPrimal(const trrom::Vector<double> & input_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & getNewPrimal() const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & getOldPrimal() const;

    void setNewGradient(const trrom::Vector<double> & input_);
    void setOldGradient(const trrom::Vector<double> & input_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & getNewGradient() const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & getOldGradient() const;

    virtual double evaluateObjective() = 0;
    virtual double evaluateObjective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_) = 0;

    virtual void computeGradient() = 0;
    virtual void computeGradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                                 const std::tr1::shared_ptr<trrom::Vector<double> > & output_) = 0;

    virtual void applyVectorToHessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                                      const std::tr1::shared_ptr<trrom::Vector<double> > & output_) = 0;

    virtual int getObjectiveFunctionEvaluationCounter() const = 0;

private:
    int m_IterationCounter;

    double m_NormTrialStep;
    double m_NormNewGradient;
    double m_StagnationMeasure;
    double m_OldObjectiveFunction;
    double m_NewObjectiveFunction;
    double m_GradientInexactnessTolerance;
    double m_ObjectiveInexactnessTolerance;

    bool m_GradientInexactnessToleranceExceeded;
    bool m_ObjectiveInexactnessToleranceExceeded;

    std::tr1::shared_ptr<trrom::Vector<double> > m_TrialStep;
    std::tr1::shared_ptr<trrom::Vector<double> > m_OldPrimal;
    std::tr1::shared_ptr<trrom::Vector<double> > m_NewPrimal;
    std::tr1::shared_ptr<trrom::Vector<double> > m_OldGradient;
    std::tr1::shared_ptr<trrom::Vector<double> > m_NewGradient;
    std::tr1::shared_ptr<trrom::Vector<double> > m_MatrixTimesVector;

private:
    void initialize(const std::tr1::shared_ptr<trrom::Data> & data_);

private:
    // unimplemented
    OptimizationDataMng(const trrom::OptimizationDataMng &);
    trrom::OptimizationDataMng & operator=(const trrom::OptimizationDataMng &);
};

}

#endif /* TRROM_OPTIMIZATIONDATAMNG_HPP_ */
