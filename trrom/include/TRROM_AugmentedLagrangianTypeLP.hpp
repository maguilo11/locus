/*
 * TRROM_AugmentedLagrangianTypeLP.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_AUGMENTEDLAGRANGIANTYPELP_HPP_
#define TRROM_AUGMENTEDLAGRANGIANTYPELP_HPP_

#include <vector>
#include "TRROM_AugmentedLagrangianAssemblyMng.hpp"

namespace trrom
{

class Data;
class ObjectiveTypeLP;
class InequalityTypeLP;

template<typename ScalarType>
class Vector;

class AugmentedLagrangianTypeLP : public trrom::AugmentedLagrangianAssemblyMng
{
public:
    AugmentedLagrangianTypeLP(const std::tr1::shared_ptr<trrom::Data> & input_,
                              const std::tr1::shared_ptr<trrom::ObjectiveTypeLP> & objective_,
                              const std::vector<std::tr1::shared_ptr<trrom::InequalityTypeLP> > & inequality_);
    virtual ~AugmentedLagrangianTypeLP();

    int getHessianCounter() const;
    void updateHessianCounter();
    int getGradientCounter() const;
    void updateGradientCounter();
    int getObjectiveCounter() const;
    void updateObjectiveCounter();
    int getInequalityCounter() const;
    void updateInequalityCounter();
    int getInequalityGradientCounter() const;
    void updateInequalityGradientCounter();

    double getPenalty() const;
    double getNormLagrangianGradient() const;
    double getNormInequalityConstraints() const;

    bool updateLagrangeMultipliers();
    void updateInequalityConstraintValues();

    double objective(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                     const double & tolerance_,
                     bool & inexactness_violated_);
    void gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                  const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_,
                  const double & tolerance_,
                  bool & inexactness_violated_);
    void hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                 const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                 const std::tr1::shared_ptr<trrom::Vector<double> > & hessian_times_vec_,
                 const double & tolerance_,
                 bool & inexactness_violated_);

    void setMinPenalty(double input_);
    void setPenaltyScaling(double input_);

private:
    int m_HessianCounter;
    int m_GradientCounter;
    int m_ObjectiveCounter;
    int m_InequalityCounter;
    int m_InequalityGradientCounter;

    double m_Penalty;
    double m_MinPenalty;
    double m_PenaltyScaling;
    double m_NormLagrangianGradient;
    double m_NormInequalityConstraints;

    std::tr1::shared_ptr<trrom::Vector<double> > m_ControlWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LagrangianGradient;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LagrangeMultipliers;
    std::tr1::shared_ptr<trrom::Vector<double> > m_InequalityConstraintValues;
    std::tr1::shared_ptr<trrom::Vector<double> > m_CurrentInequalityConstraintValues;

    std::tr1::shared_ptr<trrom::ObjectiveTypeLP> m_Objective;
    std::vector<std::tr1::shared_ptr<trrom::InequalityTypeLP> > m_Inequality;

private:
    AugmentedLagrangianTypeLP(const trrom::AugmentedLagrangianTypeLP &);
    trrom::AugmentedLagrangianTypeLP & operator=(const trrom::AugmentedLagrangianTypeLP &);
};

}

#endif /* TRROM_AUGMENTEDLAGRANGIANTYPELP_HPP_ */
