/*
 * TRROM_KelleySachs.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_INCLUDE_KELLEYSACHS_HPP_
#define TRROM_INCLUDE_KELLEYSACHS_HPP_

#include "TRROM_TrustRegionKelleySachs.hpp"

namespace trrom
{

class Data;
class KelleySachsStepMng;
class SteihaugTointDataMng;
class SteihaugTointNewtonIO;
class ProjectedSteihaugTointPcg;

template<typename ScalarType>
class Vector;

class KelleySachs : public trrom::TrustRegionKelleySachs
{
public:
    KelleySachs(const std::tr1::shared_ptr<trrom::Data> & data_,
                const std::tr1::shared_ptr<trrom::SteihaugTointDataMng> & data_mng_,
                const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_mng);
    virtual ~KelleySachs();

    void printDiagnostics();
    void setMaxNumSolverItr(int input_);

    void getMin();

private:
    void updateNumOptimizationItrDone(const int & input_);

private:
    std::tr1::shared_ptr<trrom::Vector<double> > m_MidGradient;

    std::tr1::shared_ptr<trrom::SteihaugTointNewtonIO> m_IO;
    std::tr1::shared_ptr<trrom::KelleySachsStepMng> m_StepMng;
    std::tr1::shared_ptr<trrom::SteihaugTointDataMng> m_DataMng;
    std::tr1::shared_ptr<trrom::ProjectedSteihaugTointPcg> m_Solver;

private:
    KelleySachs(const trrom::KelleySachs &);
    trrom::KelleySachs & operator=(const trrom::KelleySachs & rhs_);
};

}

#endif /* TRROM_INCLUDE_KELLEYSACHS_HPP_ */
