/*
 * TRROM_ReducedBasisData.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASISDATA_HPP_
#define TRROM_REDUCEDBASISDATA_HPP_

#include "TRROM_Data.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class ReducedBasisData : public trrom::Data
{
public:
    ReducedBasisData();
    virtual ~ReducedBasisData();

    void allocateReducedDualSolution(const trrom::Vector<double> & input_);
    void allocateDualSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateDualOrthonormalBasis(const trrom::Matrix<double> & input_);
    void allocateReducedDualLeftHandSide(const trrom::Matrix<double> & input_);
    void allocateReducedDualRightHandSide(const trrom::Vector<double> & input_);

    void allocateReducedStateSolution(const trrom::Vector<double> & input_);
    void allocateStateSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateStateOrthonormalBasis(const trrom::Matrix<double> & input_);
    void allocateReducedStateLeftHandSide(const trrom::Matrix<double> & input_);
    void allocateReducedStateRightHandSide(const trrom::Vector<double> & input_);

    void allocateLeftHandSideSnapshot(const trrom::Vector<double> & input_);
    void allocateRightHandSideSnapshot(const trrom::Vector<double> & input_);
    void allocateLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateLeftHandSideOrthonormalBasis(const trrom::Matrix<double> & input_);
    void allocateLeftHandSideDeimCoefficients(const trrom::Vector<double> & input_);

    std::tr1::shared_ptr<trrom::Vector<double> > createReducedDualSolutionCopy() const;
    std::tr1::shared_ptr<trrom::Vector<double> > createReducedDualSolutionCopy(int global_dim_) const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createDualOrthonormalBasisCopy() const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createDualOrthonormalBasisCopy(int global_num_rows_, int global_num_cols_) const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createReducedDualLeftHandSideCopy() const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createReducedDualLeftHandSideCopy(int global_num_rows_, int global_num_cols_) const;
    std::tr1::shared_ptr<trrom::Vector<double> > createReducedDualRightHandSideCopy() const;
    std::tr1::shared_ptr<trrom::Vector<double> > createReducedDualRightHandSideCopy(int global_dim_) const;

    std::tr1::shared_ptr<trrom::Vector<double> > createReducedStateSolutionCopy() const;
    std::tr1::shared_ptr<trrom::Vector<double> > createReducedStateSolutionCopy(int global_dim_) const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createStateOrthonormalBasisCopy() const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createStateOrthonormalBasisCopy(int global_num_rows_, int global_num_cols_) const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createReducedStateLeftHandSideCopy() const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createReducedStateLeftHandSideCopy(int global_num_rows_, int global_num_cols_) const;
    std::tr1::shared_ptr<trrom::Vector<double> > createReducedStateRightHandSideCopy() const;
    std::tr1::shared_ptr<trrom::Vector<double> > createReducedStateRightHandSideCopy(int global_dim_) const;

    std::tr1::shared_ptr<trrom::Vector<double> > createLeftHandSideSnapshotCopy() const;
    std::tr1::shared_ptr<trrom::Vector<double> > createLeftHandSideSnapshotCopy(int global_dim_) const;
    std::tr1::shared_ptr<trrom::Vector<double> > createRightHandSideSnapshotCopy() const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createLeftHandSideOrthonormalBasisCopy() const;
    std::tr1::shared_ptr<trrom::Matrix<double> > createLeftHandSideOrthonormalBasisCopy(int global_num_rows_, int global_num_cols_) const;
    std::tr1::shared_ptr<trrom::Vector<double> > createLeftHandSideDeimCoefficientsCopy() const;
    std::tr1::shared_ptr<trrom::Vector<double> > createLeftHandSideDeimCoefficientsCopy(int global_dim_) const;

private:
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedDualSolution;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_DualSnapshotEnsemble;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_DualOrthonormalBasis;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_ReducedDualLeftHandSide;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedDualRightHandSide;

    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedStateSolution;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_StateSnapshotEnsemble;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_StateOrthonormalBasis;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_ReducedStateLeftHandSide;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ReducedStateRightHandSide;

    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideSnapshot;
    std::tr1::shared_ptr<trrom::Vector<double> > m_RightHandSideSnapshot;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideSnapshotEnsemble;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideOrthonormalBasis;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideDeimCoefficients;

private:
    ReducedBasisData(const trrom::ReducedBasisData &);
    trrom::ReducedBasisData & operator=(const trrom::ReducedBasisData &);
};

}

#endif /* REDUCEDBASISDATAMNG_HPP_ */
