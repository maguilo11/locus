/*
 * TRROM_AssemblyManager.hpp
 *
 *  Created on: Aug 10, 2016
 */

#ifndef TRROM_ASSEMBLYMANAGER_HPP_
#define TRROM_ASSEMBLYMANAGER_HPP_

#include "TRROM_Vector.hpp"

namespace trrom
{

class AssemblyManager
{
public:
    virtual ~AssemblyManager()
    {
    }

    virtual int getHessianCounter() const = 0;
    virtual void updateHessianCounter() = 0;
    virtual int getGradientCounter() const = 0;
    virtual void updateGradientCounter() = 0;
    virtual int getObjectiveCounter() const = 0;
    virtual void updateObjectiveCounter() = 0;

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
};

}

#endif /* TRROM_ASSEMBLYMANAGER_HPP_ */
