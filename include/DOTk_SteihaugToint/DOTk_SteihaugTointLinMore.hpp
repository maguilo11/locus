/*
 * DOTk_SteihaugTointLinMore.hpp
 *
 *  Created on: Apr 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTLINMORE_HPP_
#define DOTK_STEIHAUGTOINTLINMORE_HPP_

#include <tr1/memory>
#include "DOTk_SteihaugTointNewton.hpp"

namespace dotk
{

class DOTk_SteihaugTointPcg;
class DOTk_TrustRegionStepMng;
class DOTk_SteihaugTointNewtonIO;
class DOTk_OptimizationDataMng;

class DOTk_SteihaugTointLinMore : public dotk::DOTk_SteihaugTointNewton
{
public:
    DOTk_SteihaugTointLinMore(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                              const std::tr1::shared_ptr<dotk::DOTk_TrustRegionStepMng> & step_mng);
    virtual ~DOTk_SteihaugTointLinMore();

    void setSolverMaxNumItr(size_t input_);
    size_t getSolverMaxNumItr() const;
    void setSolverRelativeTolerance(Real input_);
    Real getSolverRelativeTolerance() const;
    void setSolverRelativeToleranceExponential(Real input_);
    Real getSolverRelativeToleranceExponential() const;

    void printDiagnosticsAndSolutionAtEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    virtual void getMin();
    virtual void updateNumOptimizationItrDone(const size_t & input_);

private:
    bool checkStoppingCriteria();
    void resetCurrentStateToPreviousState();

private:
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointPcg> m_Solver;
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionStepMng> m_StepMng;
    std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> m_DataMng;

private:
    DOTk_SteihaugTointLinMore(const dotk::DOTk_SteihaugTointLinMore &);
    dotk::DOTk_SteihaugTointLinMore & operator=(const dotk::DOTk_SteihaugTointLinMore & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTLINMORE_HPP_ */
