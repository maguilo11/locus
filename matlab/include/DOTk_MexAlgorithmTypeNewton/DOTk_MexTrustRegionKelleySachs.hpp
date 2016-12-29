/*
 * DOTk_MexTrustRegionKelleySachs.hpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXTRUSTREGIONKELLEYSACHS_HPP_
#define DOTK_MEXTRUSTREGIONKELLEYSACHS_HPP_

#include "DOTk_Types.hpp"
#include "DOTk_MexSteihaugTointNewton.hpp"

namespace dotk
{

class DOTk_SteihaugTointKelleySachs;

class DOTk_MexTrustRegionKelleySachs : public dotk::DOTk_MexSteihaugTointNewton
{
public:
    explicit DOTk_MexTrustRegionKelleySachs(const mxArray* options_[]);
    virtual ~DOTk_MexTrustRegionKelleySachs();

    size_t getMaxNumUpdates() const;
    size_t getMaxNumSteihaugTointSolverItr() const;

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initializeKelleySachsTrustRegion(const mxArray* options_[]);
    void setKelleySachsAlgorithmParameters(dotk::DOTk_SteihaugTointKelleySachs & algorithm_);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

private:
    dotk::types::problem_t m_ProblemType;

    size_t m_MaxNumUpdates;
    size_t m_MaxNumSteihaugTointSolverItr;

    mxArray* m_ObjectiveFunction;
    mxArray* m_EqualityConstraint;

private:
    DOTk_MexTrustRegionKelleySachs(const dotk::DOTk_MexTrustRegionKelleySachs & rhs_);
    dotk::DOTk_MexTrustRegionKelleySachs& operator=(const dotk::DOTk_MexTrustRegionKelleySachs & rhs_);
};

}

#endif /* DOTK_MEXTRUSTREGIONKELLEYSACHS_HPP_ */
