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

class ReducedBasisData : public trrom::Data
{
public:
    ReducedBasisData();
    virtual ~ReducedBasisData();

    void allocateLeftHandSideSnapshot(const trrom::Vector<double> & input_);
    void allocateRightHandSideSnapshot(const trrom::Vector<double> & input_);

    trrom::types::fidelity_t fidelity() const;
    void fidelity(trrom::types::fidelity_t input_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & getLeftHandSideSnapshot() const;
    void setLeftHandSideSnapshot(const trrom::Vector<double> & input_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & getRightHandSideSnapshot() const;
    void setRightHandSideSnapshot(const trrom::Vector<double> & input_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & getLeftHandSideActiveIndices() const;
    void setLeftHandSideActiveIndices(const trrom::Vector<double> & input_);

private:
    trrom::types::fidelity_t m_Fidelity;

    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideSnapshot;
    std::tr1::shared_ptr<trrom::Vector<double> > m_RightHandSideSnapshot;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideActiveIndices;

private:
    ReducedBasisData(const trrom::ReducedBasisData &);
    trrom::ReducedBasisData & operator=(const trrom::ReducedBasisData &);
};

}

#endif /* REDUCEDBASISDATAMNG_HPP_ */
