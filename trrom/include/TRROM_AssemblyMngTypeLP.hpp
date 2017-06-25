/*
 * TRROM_AssemblyMngTypeLP.hpp
 *
 *  Created on: Aug 10, 2016
 */

#ifndef TRROM_ASSEMBLYMNGTYPELP_HPP_
#define TRROM_ASSEMBLYMNGTYPELP_HPP_

#include "TRROM_AssemblyManager.hpp"

namespace trrom
{

class ObjectiveTypeLP;

template<typename ScalarType>
class Vector;

class AssemblyMngTypeLP : public trrom::AssemblyManager
{
public:
    explicit AssemblyMngTypeLP(const std::shared_ptr<trrom::ObjectiveTypeLP> & objective_);
    virtual ~AssemblyMngTypeLP();

    virtual int getHessianCounter() const;
    virtual void updateHessianCounter();
    virtual int getGradientCounter() const;
    virtual void updateGradientCounter();
    virtual int getObjectiveCounter() const;
    virtual void updateObjectiveCounter();

    virtual double objective(const std::shared_ptr<trrom::Vector<double> > & input_,
                             const double & tolerance_,
                             bool & inexactness_violated_);
    virtual void gradient(const std::shared_ptr<trrom::Vector<double> > & input_,
                          const std::shared_ptr<trrom::Vector<double> > & gradient_,
                          const double & tolerance_,
                          bool & inexactness_violated_);
    virtual void hessian(const std::shared_ptr<trrom::Vector<double> > & input_,
                         const std::shared_ptr<trrom::Vector<double> > & vector_,
                         const std::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                         const double & tolerance_,
                         bool & inexactness_violated_);

private:
    int m_HessianCounter;
    int m_GradientCounter;
    int m_ObjectiveCounter;

    std::shared_ptr<trrom::ObjectiveTypeLP> m_Objective;

private:
    AssemblyMngTypeLP(const trrom::AssemblyMngTypeLP &);
    trrom::AssemblyMngTypeLP & operator=(const trrom::AssemblyMngTypeLP &);
};

}

#endif /* TRROM_ASSEMBLYMNGTYPELP_HPP_ */
