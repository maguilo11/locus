/*
 * DOTk_SteihaugTointKelleySachs.hpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTKELLEYSACHS_HPP_
#define DOTK_STEIHAUGTOINTKELLEYSACHS_HPP_

#include <tr1/memory>
#include "DOTk_SteihaugTointNewton.hpp"

namespace dotk
{

class DOTk_BoundConstraints;
class DOTk_KelleySachsStepMng;
class DOTk_OptimizationDataMng;
class DOTk_SteihaugTointNewtonIO;
class DOTk_ProjectedSteihaugTointPcg;

template<typename ScalarType>
class Vector;

class DOTk_SteihaugTointKelleySachs : public dotk::DOTk_SteihaugTointNewton
{
public:
    DOTk_SteihaugTointKelleySachs(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                  const std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> & step_mng);
    virtual ~DOTk_SteihaugTointKelleySachs();

    void setMaxNumUpdates(size_t input_);
    size_t getMaxNumUpdates() const;
    void setMaxNumSolverItr(size_t input_);

    void printDiagnosticsAndSolutionAtEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    virtual void getMin();
    virtual void updateNumOptimizationItrDone(const size_t & input_);

private:
    bool updatePrimal();
    void updateDataManager();
    bool checkStoppingCriteria();
    void computeStationarityMeasure();
    void resetCurrentStateToPreviousState();

private:
    size_t m_MaxNumUpdates;
    Real m_StationarityMeasure;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_WorkVector;

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> m_StepMng;
    std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> m_DataMng;
    std::tr1::shared_ptr<dotk::DOTk_BoundConstraints> m_BoundConstraint;
    std::tr1::shared_ptr<dotk::DOTk_ProjectedSteihaugTointPcg> m_Solver;

private:
    DOTk_SteihaugTointKelleySachs(const dotk::DOTk_SteihaugTointKelleySachs &);
    dotk::DOTk_SteihaugTointKelleySachs & operator=(const dotk::DOTk_SteihaugTointKelleySachs & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTKELLEYSACHS_HPP_ */
