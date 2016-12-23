/*
 * TRROM_InexactNewtonDataMng.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_INEXACTNEWTONDATAMNG_HPP_
#define TRROM_INEXACTNEWTONDATAMNG_HPP_

#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

class Data;
class AssemblyManager;

template<typename ScalarType>
class Vector;

class InexactNewtonDataMng : public trrom::OptimizationDataMng
{
public:
    InexactNewtonDataMng(const std::tr1::shared_ptr<trrom::Data> & data_,
                         const std::tr1::shared_ptr<trrom::AssemblyManager> & manager_);
    virtual ~InexactNewtonDataMng();

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
    InexactNewtonDataMng(const trrom::InexactNewtonDataMng &);
    trrom::InexactNewtonDataMng & operator=(const trrom::InexactNewtonDataMng & rhs_);
};

}

#endif /* TRROM_INEXACTNEWTONDATAMNG_HPP_ */
