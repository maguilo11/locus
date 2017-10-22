/*
 * Locus_LineSearch.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_LINESEARCH_HPP_
#define LOCUS_LINESEARCH_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class StateManager;

template<typename ScalarType, typename OrdinalType = size_t>
class LineSearch
{
public:
    virtual ~LineSearch()
    {
    }

    virtual OrdinalType getNumIterationsDone() const = 0;
    virtual void setMaxNumIterations(const OrdinalType & aInput) = 0;
    virtual void setContractionFactor(const ScalarType & aInput) = 0;
    virtual void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng) = 0;
};

} // namespace locus

#endif /* LOCUS_LINESEARCH_HPP_ */
