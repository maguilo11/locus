/*
 * TRROM_KelleySachsReducedBasis.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_KELLEYSACHSREDUCEDBASIS_HPP_
#define TRROM_KELLEYSACHSREDUCEDBASIS_HPP_

#include "TRROM_TrustRegionKelleySachs.hpp"

namespace trrom
{

class ReducedBasisData;
class KelleySachsStepMng;
class ReducedBasisDataMng;
class SteihaugTointNewtonIO;
class ProjectedSteihaugTointPcg;

template<typename ScalarType>
class Vector;

class KelleySachsReducedBasis : public trrom::TrustRegionKelleySachs
{
public:
    KelleySachsReducedBasis(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                            const std::tr1::shared_ptr<trrom::ReducedBasisDataMng> & data_mng_,
                            const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_mng);
    virtual ~KelleySachsReducedBasis();

    void printDiagnostics();
    void setMaxNumSolverItr(int input_);

    void getMin();

private:
    void solveSubProblem();
    void updateNumOptimizationItrDone(const int & input_);

private:
    std::tr1::shared_ptr<trrom::Vector<double> > m_MidGradient;

    std::tr1::shared_ptr<trrom::SteihaugTointNewtonIO> m_IO;
    std::tr1::shared_ptr<trrom::KelleySachsStepMng> m_StepMng;
    std::tr1::shared_ptr<trrom::ReducedBasisDataMng> m_DataMng;
    std::tr1::shared_ptr<trrom::ProjectedSteihaugTointPcg> m_Solver;

private:
    KelleySachsReducedBasis(const trrom::KelleySachsReducedBasis &);
    trrom::KelleySachsReducedBasis & operator=(const trrom::KelleySachsReducedBasis & rhs_);
};

}

#endif /* TRROM_KELLEYSACHSREDUCEDBASIS_HPP_ */
