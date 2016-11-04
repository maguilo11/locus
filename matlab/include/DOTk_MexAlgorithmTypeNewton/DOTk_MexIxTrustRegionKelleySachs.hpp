/*
 * DOTk_MexIxTrustRegionKelleySachs.hpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXIXTRUSTREGIONKELLEYSACHS_HPP_
#define DOTK_MEXIXTRUSTREGIONKELLEYSACHS_HPP_

#include "DOTk_MexSteihaugTointNewton.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;
class DOTk_SteihaugTointKelleySachs;

class DOTk_MexIxTrustRegionKelleySachs : public dotk::DOTk_MexSteihaugTointNewton
{
public:
    explicit DOTk_MexIxTrustRegionKelleySachs(const mxArray* options_[]);
    virtual ~DOTk_MexIxTrustRegionKelleySachs();

    size_t getMaxNumUpdates() const;
    size_t getMaxNumSteihaugTointSolverItr() const;

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initializeIxKelleySachsTrustRegion(const mxArray* options_[]);
    void setIxKelleySachsAlgorithmParameters(dotk::DOTk_SteihaugTointKelleySachs & algorithm_);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

private:
    dotk::types::problem_t m_ProblemType;

    size_t m_MaxNumUpdates;
    size_t m_MaxNumSteihaugTointSolverItr;

    dotk::DOTk_MexArrayPtr m_ObjectiveFunctionOperators;
    dotk::DOTk_MexArrayPtr m_EqualityConstraintOperators;

private:
    DOTk_MexIxTrustRegionKelleySachs(const dotk::DOTk_MexIxTrustRegionKelleySachs & rhs_);
    dotk::DOTk_MexIxTrustRegionKelleySachs& operator=(const dotk::DOTk_MexIxTrustRegionKelleySachs & rhs_);
};

}

#endif /* DOTK_MEXIXTRUSTREGIONKELLEYSACHS_HPP_ */
