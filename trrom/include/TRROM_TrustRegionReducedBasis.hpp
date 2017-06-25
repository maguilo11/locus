/*
 * TRROM_TrustRegionReducedBasis.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TRUSTREGIONREDUCEDBASIS_HPP_
#define TRROM_TRUSTREGIONREDUCEDBASIS_HPP_

#include "TRROM_TrustRegionNewtonBase.hpp"

namespace trrom
{

class ReducedBasisData;
class KelleySachsStepMng;
class TrustRegionNewtonIO;
class ProjectedSteihaugTointPcg;
class ReducedBasisNewtonDataMng;

template<typename ScalarType>
class Vector;

class TrustRegionReducedBasis : public trrom::TrustRegionNewtonBase
{
public:
    TrustRegionReducedBasis(const std::shared_ptr<trrom::ReducedBasisData> & data_,
                            const std::shared_ptr<trrom::KelleySachsStepMng> & step_mng,
                            const std::shared_ptr<trrom::ReducedBasisNewtonDataMng> & data_mng_);
    virtual ~TrustRegionReducedBasis();

    void printDiagnostics();
    void setMaxNumSolverItr(int input_);

    void getMin();

private:
    void solveSubProblem();
    void updateNumOptimizationItrDone(const int & input_);

private:
    std::shared_ptr<trrom::Vector<double> > m_MidGradient;

    std::shared_ptr<trrom::TrustRegionNewtonIO> m_IO;
    std::shared_ptr<trrom::KelleySachsStepMng> m_StepMng;
    std::shared_ptr<trrom::ProjectedSteihaugTointPcg> m_Solver;
    std::shared_ptr<trrom::ReducedBasisNewtonDataMng> m_DataMng;

private:
    TrustRegionReducedBasis(const trrom::TrustRegionReducedBasis &);
    trrom::TrustRegionReducedBasis & operator=(const trrom::TrustRegionReducedBasis & rhs_);
};

}

#endif /* TRROM_TRUSTREGIONREDUCEDBASIS_HPP_ */
