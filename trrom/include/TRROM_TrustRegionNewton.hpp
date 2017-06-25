/*
 * TRROM_TrustRegionNewton.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TRUSTREGIONNEWTON_HPP_
#define TRROM_TRUSTREGIONNEWTON_HPP_

#include "TRROM_TrustRegionNewtonBase.hpp"

namespace trrom
{

class Data;
class KelleySachsStepMng;
class TrustRegionNewtonIO;
class InexactNewtonDataMng;
class ProjectedSteihaugTointPcg;

template<typename ScalarType>
class Vector;

class TrustRegionNewton : public trrom::TrustRegionNewtonBase
{
public:
    TrustRegionNewton(const std::shared_ptr<trrom::Data> & data_,
                      const std::shared_ptr<trrom::KelleySachsStepMng> & step_mng,
                      const std::shared_ptr<trrom::InexactNewtonDataMng> & data_mng_);
    virtual ~TrustRegionNewton();

    void printDiagnostics();
    void setMaxNumSolverItr(int input_);

    void getMin();

private:
    void updateNumOptimizationItrDone(const int & input_);

private:
    std::shared_ptr<trrom::Vector<double> > m_MidGradient;

    std::shared_ptr<trrom::TrustRegionNewtonIO> m_IO;
    std::shared_ptr<trrom::KelleySachsStepMng> m_StepMng;
    std::shared_ptr<trrom::InexactNewtonDataMng> m_DataMng;
    std::shared_ptr<trrom::ProjectedSteihaugTointPcg> m_Solver;

private:
    TrustRegionNewton(const trrom::TrustRegionNewton &);
    trrom::TrustRegionNewton & operator=(const trrom::TrustRegionNewton & rhs_);
};

}

#endif /* TRROM_TRUSTREGIONNEWTON_HPP_ */
