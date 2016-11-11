/*
 * TRROM_ReducedBasisDataMng.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_REDUCEDBASISDATAMNG_HPP_
#define TRROM_REDUCEDBASISDATAMNG_HPP_

#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

class ReducedBasisData;
class ReducedBasisAssemblyMng;

template<typename ScalarType>
class Vector;

class ReducedBasisDataMng : public trrom::OptimizationDataMng
{
public:
    ReducedBasisDataMng(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                        const std::tr1::shared_ptr<trrom::ReducedBasisAssemblyMng> & manager_);
    virtual ~ReducedBasisDataMng();

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
    ReducedBasisDataMng(const trrom::ReducedBasisDataMng &);
    trrom::ReducedBasisDataMng & operator=(const trrom::ReducedBasisDataMng &);
};

}

#endif /* TRROM_REDUCEDBASISDATAMNG_HPP_ */
