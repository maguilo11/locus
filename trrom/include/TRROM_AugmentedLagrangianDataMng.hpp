/*
 * TRROM_AugmentedLagrangianDataMng.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_AUGMENTEDLAGRANGIANDATAMNG_HPP_
#define TRROM_AUGMENTEDLAGRANGIANDATAMNG_HPP_

#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

class Data;
class AugmentedLagrangianAssemblyMng;

template<typename ScalarType>
class Vector;

class AugmentedLagrangianDataMng : public trrom::OptimizationDataMng
{
public:
    AugmentedLagrangianDataMng(const std::tr1::shared_ptr<trrom::Data> & data_,
                               const std::tr1::shared_ptr<trrom::AugmentedLagrangianAssemblyMng> & mng_);
    virtual ~AugmentedLagrangianDataMng();

    double evaluateObjective();
    double evaluateObjective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_);
    int getObjectiveFunctionEvaluationCounter() const;

    void computeGradient();
    void computeGradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & output_);

    void applyVectorToHessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                              const std::tr1::shared_ptr<trrom::Vector<double> > & output_);

    double getPenalty() const;
    double getNormInequalityConstraints() const;
    double getNormLagrangianGradient() const;

    bool updateLagrangeMultipliers();
    void updateInequalityConstraintValues();

private:
    std::tr1::shared_ptr<trrom::AugmentedLagrangianAssemblyMng> m_AssemblyMng;

private:
    AugmentedLagrangianDataMng(const trrom::AugmentedLagrangianDataMng &);
    trrom::AugmentedLagrangianDataMng & operator=(const trrom::AugmentedLagrangianDataMng & rhs_);
};

}

#endif /* TRROM_AUGMENTEDLAGRANGIANDATAMNG_HPP_ */
