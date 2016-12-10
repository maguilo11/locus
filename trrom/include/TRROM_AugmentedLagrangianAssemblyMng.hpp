/*
 * TRROM_AugmentedLagrangianAssemblyMng.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_AUGMENTEDLAGRANGIANASSEMBLYMNG_HPP_
#define TRROM_AUGMENTEDLAGRANGIANASSEMBLYMNG_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class AugmentedLagrangianAssemblyMng
{
public:
    virtual ~AugmentedLagrangianAssemblyMng()
    {
    }

    virtual int getHessianCounter() const = 0;
    virtual void updateHessianCounter() = 0;
    virtual int getGradientCounter() const = 0;
    virtual void updateGradientCounter() = 0;
    virtual int getObjectiveCounter() const = 0;
    virtual void updateObjectiveCounter() = 0;
    virtual int getInequalityCounter() const = 0;
    virtual void updateInequalityCounter() = 0;
    virtual int getInequalityGradientCounter() const = 0;
    virtual void updateInequalityGradientCounter() = 0;

    virtual double getPenalty() const = 0;
    virtual double getNormLagrangianGradient() const = 0;
    virtual double getNormInequalityConstraints() const = 0;

    virtual void updateInequalityConstraintValues() = 0;
    virtual double objective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                             const double & tolerance_,
                             bool & inexactness_violated_) = 0;
    virtual void gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                          const std::tr1::shared_ptr<trrom::Vector<double> > & output_,
                          const double & tolerance_,
                          bool & inexactness_violated_) = 0;
    virtual void hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                         const std::tr1::shared_ptr<trrom::Vector<double> > & output_,
                         const double & tolerance_,
                         bool & inexactness_violated_) = 0;
    virtual bool updateLagrangeMultipliers() = 0;
};

}

#endif /* TRROM_AUGMENTEDLAGRANGIANASSEMBLYMNG_HPP_ */
