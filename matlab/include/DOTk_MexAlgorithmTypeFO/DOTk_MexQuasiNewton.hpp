/*
 * DOTk_MexQuasiNewton.hpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXQUASINEWTON_HPP_
#define DOTK_MEXQUASINEWTON_HPP_

#include "DOTk_Types.hpp"
#include "DOTk_MexAlgorithmTypeFO.hpp"

namespace dotk
{

class DOTk_LineSearchStepMng;
class DOTk_LineSearchQuasiNewton;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_MexQuasiNewton : public DOTk_MexAlgorithmTypeFO
{
public:
    explicit DOTk_MexQuasiNewton(const mxArray* options_[]);
    virtual ~DOTk_MexQuasiNewton();

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

    void setAlgorithmParameters(dotk::DOTk_LineSearchQuasiNewton & algorithm_);
    void optimize(const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                  const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_,
                  const mxArray* input_[],
                  mxArray* output_[]);

private:
    mxArray* m_ObjectiveFunction;
    mxArray* m_EqualityConstraint;

private:
    DOTk_MexQuasiNewton(const dotk::DOTk_MexQuasiNewton & rhs_);
    dotk::DOTk_MexQuasiNewton& operator=(const dotk::DOTk_MexQuasiNewton & rhs_);
};

}

#endif /* DOTK_MEXQUASINEWTON_HPP_ */
