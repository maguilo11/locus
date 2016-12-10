/*
 * SteihaugTointDataMng.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_STEIHAUGTOINTDATAMNG_HPP_
#define TRROM_STEIHAUGTOINTDATAMNG_HPP_

#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

class Data;
class AssemblyManager;

template<typename ScalarType>
class Vector;

class SteihaugTointDataMng : public trrom::OptimizationDataMng
{
public:
    SteihaugTointDataMng(const std::tr1::shared_ptr<trrom::Data> & data_,
                         const std::tr1::shared_ptr<trrom::AssemblyManager> & manager_);
    virtual ~SteihaugTointDataMng();

    double evaluateObjective();
    double evaluateObjective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_);
    int getObjectiveFunctionEvaluationCounter() const;

    void computeGradient();
    void computeGradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & output_);

    void applyVectorToHessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                              const std::tr1::shared_ptr<trrom::Vector<double> > & output_);

    const std::tr1::shared_ptr<trrom::Data> & getData() const;

private:
    std::tr1::shared_ptr<trrom::Data> m_Data;
    std::tr1::shared_ptr<trrom::AssemblyManager> m_AssemblyMng;

private:
    SteihaugTointDataMng(const trrom::SteihaugTointDataMng &);
    trrom::SteihaugTointDataMng & operator=(const trrom::SteihaugTointDataMng & rhs_);
};

}

#endif /* TRROM_STEIHAUGTOINTDATAMNG_HPP_ */
