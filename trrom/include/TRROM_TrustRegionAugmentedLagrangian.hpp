/*
 * TRROM_TrustRegionAugmentedLagrangian.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TRUSTREGIONAUGMENTEDLAGRANGIAN_HPP_
#define TRROM_TRUSTREGIONAUGMENTEDLAGRANGIAN_HPP_

#include "TRROM_TrustRegionNewtonBase.hpp"

namespace trrom
{

class Data;
class KelleySachsStepMng;
class TrustRegionNewtonIO;
class ProjectedSteihaugTointPcg;
class AugmentedLagrangianDataMng;

template<typename ScalarType>
class Vector;

class TrustRegionAugmentedLagrangian : public trrom::TrustRegionNewtonBase
{
public:
    TrustRegionAugmentedLagrangian(const std::shared_ptr<trrom::Data> & data_,
                                   const std::shared_ptr<trrom::KelleySachsStepMng> & step_mng,
                                   const std::shared_ptr<trrom::AugmentedLagrangianDataMng> & data_mng_);
    virtual ~TrustRegionAugmentedLagrangian();

    void printDiagnostics();
    void setOptimalityTolerance(double input_);
    void setFeasibilityTolerance(double input_);

    void getMin();

private:
    bool checkNaN();
    void updateDataManager();
    bool checkStoppingCriteria();
    bool checkPrimaryStoppingCriteria();

    void updateNumOptimizationItrDone(const int & input_);

private:
    double m_Gamma;
    double m_OptimalityTolerance;
    double m_FeasibilityTolerance;

    std::shared_ptr<trrom::Vector<double> > m_Vector;
    std::shared_ptr<trrom::Vector<double> > m_MidGradient;

    std::shared_ptr<trrom::TrustRegionNewtonIO> m_IO;
    std::shared_ptr<trrom::KelleySachsStepMng> m_StepMng;
    std::shared_ptr<trrom::ProjectedSteihaugTointPcg> m_Solver;
    std::shared_ptr<trrom::AugmentedLagrangianDataMng> m_DataMng;

private:
    TrustRegionAugmentedLagrangian(const trrom::TrustRegionAugmentedLagrangian &);
    trrom::TrustRegionAugmentedLagrangian & operator=(const trrom::TrustRegionAugmentedLagrangian & rhs_);
};

}

#endif /* TRROM_TRUSTREGIONAUGMENTEDLAGRANGIAN_HPP_ */
