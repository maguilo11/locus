/*
 * DOTk_MexInexactTrustRegionSQP.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXINEXACTTRUSTREGIONSQP_HPP_
#define DOTK_MEXINEXACTTRUSTREGIONSQP_HPP_

#include "DOTk_MexAlgorithmTypeSQP.hpp"

namespace dotk
{

class DOTk_InexactTrustRegionSQP;
class DOTk_TrustRegionMngTypeELP;
class DOTk_InexactTrustRegionSqpSolverMng;

class DOTk_MexInexactTrustRegionSQP : public dotk::DOTk_MexAlgorithmTypeSQP
{
public:
    explicit DOTk_MexInexactTrustRegionSQP(const mxArray* options_[]);
    ~DOTk_MexInexactTrustRegionSQP();

    void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);

    void setTrustRegionParameters(const mxArray* options_, dotk::DOTk_TrustRegionMngTypeELP & mng_);
    void setAlgorithmParameters(const mxArray* options_, dotk::DOTk_InexactTrustRegionSQP & algorithm_);
    void setSqpKrylovSolversParameters(const mxArray* options_, dotk::DOTk_InexactTrustRegionSqpSolverMng & mng_);

    void solveTypeEqualityConstrainedLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeEqualityConstrainedNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

    void gatherOutputDataTypeLP(dotk::DOTk_InexactTrustRegionSQP & algorithm_,
                                dotk::DOTk_TrustRegionMngTypeELP & mng_,
                                mxArray* output_[]);
    void gatherOutputDataTypeNLP(dotk::DOTk_InexactTrustRegionSQP & algorithm_,
                                 dotk::DOTk_TrustRegionMngTypeELP & mng_,
                                 mxArray* output_[]);

private:
    mxArray* m_ObjectiveFunction;
    mxArray* m_EqualityConstraint;

private:
    DOTk_MexInexactTrustRegionSQP(const dotk::DOTk_MexInexactTrustRegionSQP&);
    dotk::DOTk_MexInexactTrustRegionSQP& operator=(const dotk::DOTk_MexInexactTrustRegionSQP&);
};

}

#endif /* DOTK_MEXINEXACTTRUSTREGIONSQP_HPP_ */
