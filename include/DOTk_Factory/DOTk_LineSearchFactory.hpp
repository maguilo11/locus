/*
 * DOTk_LineSearchFactory.hpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHFACTORY_HPP_
#define DOTK_LINESEARCHFACTORY_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_LineSearch;
template<typename Type>
class vector;

class DOTk_LineSearchFactory
{
public:
    DOTk_LineSearchFactory();
    explicit DOTk_LineSearchFactory(dotk::types::line_search_t type_);
    ~DOTk_LineSearchFactory();

    void setFactoryType(dotk::types::line_search_t type_);
    dotk::types::line_search_t getFactoryType() const;

    void buildArmijoLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                               std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_);
    void buildGoldsteinLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                  std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_);
    void buildCubicLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                              std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_);
    void buildGoldenSectionLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                      std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_);

    void build(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
               std::tr1::shared_ptr<dotk::DOTk_LineSearch> & line_search_) const;

private:
    dotk::types::line_search_t m_FactoryType;

private:
    DOTk_LineSearchFactory(const dotk::DOTk_LineSearchFactory &);
    DOTk_LineSearchFactory operator=(const dotk::DOTk_LineSearchFactory &);
};

}

#endif /* DOTK_LINESEARCHFACTORY_HPP_ */
