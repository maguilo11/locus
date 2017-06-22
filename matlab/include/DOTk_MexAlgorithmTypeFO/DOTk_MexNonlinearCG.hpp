/*
 * DOTk_MexNonlinearCG.hpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXNONLINEARCG_HPP_
#define DOTK_MEXNONLINEARCG_HPP_

#include "DOTk_MexAlgorithmTypeFO.hpp"

namespace dotk
{

class DOTk_NonlinearCG;
class DOTk_LineSearchStepMng;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_MexNonlinearCG : public DOTk_MexAlgorithmTypeFO
{
public:
    explicit DOTk_MexNonlinearCG(const mxArray* options_[]);
    virtual ~DOTk_MexNonlinearCG();

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);
    void setAlgorithmParameters(dotk::DOTk_NonlinearCG & algorithm_);
    void setAlgorithmType(const dotk::types::nonlinearcg_t & type_, dotk::DOTk_NonlinearCG & algorithm_);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

    void optimize(const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                  const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_,
                  const mxArray* input_[],
                  mxArray* output_[]);

private:
    dotk::types::nonlinearcg_t m_NonlinearCgType;
    mxArray* m_ObjectiveFunction;
    mxArray* m_EqualityConstraint;

private:
    DOTk_MexNonlinearCG(const dotk::DOTk_MexNonlinearCG & rhs_);
    dotk::DOTk_MexNonlinearCG& operator=(const dotk::DOTk_MexNonlinearCG & rhs_);
};

}

#endif /* DOTK_MEXNONLINEARCG_HPP_ */
