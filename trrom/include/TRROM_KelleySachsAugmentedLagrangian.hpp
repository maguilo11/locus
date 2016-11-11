/*
 * TRROM_KelleySachsAugmentedLagrangian.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_KELLEYSACHSAUGMENTEDLAGRANGIAN_HPP_
#define TRROM_KELLEYSACHSAUGMENTEDLAGRANGIAN_HPP_

#include "TRROM_TrustRegionKelleySachs.hpp"

namespace trrom
{

class Data;
class KelleySachsStepMng;
class SteihaugTointNewtonIO;
class ProjectedSteihaugTointPcg;
class AugmentedLagrangianDataMng;

template<typename ScalarType>
class Vector;

class KelleySachsAugmentedLagrangian : public trrom::TrustRegionKelleySachs
{
public:
    KelleySachsAugmentedLagrangian(const std::tr1::shared_ptr<trrom::Data> & data_,
                                   const std::tr1::shared_ptr<trrom::AugmentedLagrangianDataMng> & data_mng_,
                                   const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_mng);
    virtual ~KelleySachsAugmentedLagrangian();

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

    std::tr1::shared_ptr<trrom::Vector<double> > m_WorkVector;
    std::tr1::shared_ptr<trrom::Vector<double> > m_MidGradient;

    std::tr1::shared_ptr<trrom::SteihaugTointNewtonIO> m_IO;
    std::tr1::shared_ptr<trrom::KelleySachsStepMng> m_StepMng;
    std::tr1::shared_ptr<trrom::ProjectedSteihaugTointPcg> m_Solver;
    std::tr1::shared_ptr<trrom::AugmentedLagrangianDataMng> m_DataMng;

private:
    KelleySachsAugmentedLagrangian(const trrom::KelleySachsAugmentedLagrangian &);
    trrom::KelleySachsAugmentedLagrangian & operator=(const trrom::KelleySachsAugmentedLagrangian & rhs_);
};

}

#endif /* TRROM_KELLEYSACHSAUGMENTEDLAGRANGIAN_HPP_ */
