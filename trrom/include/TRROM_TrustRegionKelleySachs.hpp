/*
 * TRROM_TrustRegionKelleySachs.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TRUSTREGIONKELLEYSACHS_HPP_
#define TRROM_TRUSTREGIONKELLEYSACHS_HPP_

namespace trrom
{

class Data;
class BoundConstraints;
class KelleySachsStepMng;
class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class TrustRegionKelleySachs
{
public:
    explicit TrustRegionKelleySachs(const std::tr1::shared_ptr<trrom::Data> & data_);
    virtual ~TrustRegionKelleySachs();

    void setGradientTolerance(double input_);
    double getGradientTolerance() const;
    void setTrialStepTolerance(double input_);
    double getTrialStepTolerance() const;
    void setObjectiveTolerance(double input_);
    double getObjectiveTolerance() const;
    void setStagnationTolerance(double input_);
    double getStagnationTolerance() const;
    void setActualReductionTolerance(double input_);
    double getActualReductionTolerance() const;
    double getStationarityMeasure() const;

    void setMaxNumUpdates(int input_);
    int getMaxNumUpdates() const;
    void setNumOptimizationItrDone(int input_);
    int getNumOptimizationItrDone() const;
    void setMaxNumOptimizationItr(int input_);
    int getMaxNumOptimizationItr() const;
    void setStoppingCriterion(trrom::types::stop_criterion_t input_);
    trrom::types::stop_criterion_t getStoppingCriterion() const;

    bool updatePrimal(const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_,
                      const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_,
                      const std::tr1::shared_ptr<trrom::Vector<double> > & mid_gradient_);
    void updateDataManager(const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_,
                           const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_,
                           const std::tr1::shared_ptr<trrom::Vector<double> > & mid_gradient_,
                           const std::tr1::shared_ptr<trrom::Vector<double> > & inactive_set_);
    bool checkStoppingCriteria(const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_,
                               const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_);
    void computeStationarityMeasure(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_,
                                    const std::tr1::shared_ptr<trrom::Vector<double> > & inactive_set_);
    void resetCurrentStateToPreviousState(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_);

    virtual void getMin() = 0;

private:
    int m_MaxNumUpdates;
    int m_MaxNumOptimizationItr;
    int m_NumOptimizationItrDone;

    double m_GradientTolerance;
    double m_TrialStepTolerance;
    double m_ObjectiveTolerance;
    double m_StagnationTolerance;
    double m_StationarityMeasure;
    double m_ActualReductionTolerance;

    trrom::types::stop_criterion_t m_StoppingCriterion;

    std::tr1::shared_ptr<trrom::Vector<double> > m_WorkVector;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LowerBound;
    std::tr1::shared_ptr<trrom::Vector<double> > m_UpperBound;
    std::tr1::shared_ptr<trrom::BoundConstraints> m_BoundConstraint;

private:
    TrustRegionKelleySachs(const trrom::TrustRegionKelleySachs &);
    trrom::TrustRegionKelleySachs & operator=(const trrom::TrustRegionKelleySachs & rhs_);
};

}

#endif /* TRROM_TRUSTREGIONKELLEYSACHS_HPP_ */
