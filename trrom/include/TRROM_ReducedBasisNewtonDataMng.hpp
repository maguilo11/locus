/*
 * TRROM_ReducedBasisNewtonDataMng.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_REDUCEDBASISNEWTONDATAMNG_HPP_
#define TRROM_REDUCEDBASISNEWTONDATAMNG_HPP_

#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

class ReducedBasisData;
class ReducedBasisAssemblyMng;

template<typename ScalarType>
class Vector;

class ReducedBasisNewtonDataMng : public trrom::OptimizationDataMng
{
public:
    ReducedBasisNewtonDataMng(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                              const std::tr1::shared_ptr<trrom::ReducedBasisAssemblyMng> & manager_);
    virtual ~ReducedBasisNewtonDataMng();

    int getObjectiveFunctionEvaluationCounter() const;

    void computeGradient();
    void computeGradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & output_);
    double evaluateObjective();
    double evaluateObjective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_);
    void applyVectorToHessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                              const std::tr1::shared_ptr<trrom::Vector<double> > & output_);

    void updateLowFidelityModel();
    trrom::types::fidelity_t fidelity() const;
    void fidelity(trrom::types::fidelity_t input_);

private:
    std::tr1::shared_ptr<trrom::ReducedBasisAssemblyMng> m_AssemblyMng;

private:
    ReducedBasisNewtonDataMng(const trrom::ReducedBasisNewtonDataMng &);
    trrom::ReducedBasisNewtonDataMng & operator=(const trrom::ReducedBasisNewtonDataMng &);
};

}

#endif /* TRROM_REDUCEDBASISNEWTONDATAMNG_HPP_ */
