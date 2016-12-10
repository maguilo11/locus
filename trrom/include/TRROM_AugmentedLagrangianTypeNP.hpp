/*
 * TRROM_AugmentedLagrangianTypeNP.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_AUGMENTEDLAGRANGIANTYPENP_HPP_
#define TRROM_AUGMENTEDLAGRANGIANTYPENP_HPP_

#include <vector>
#include "TRROM_AugmentedLagrangianAssemblyMng.hpp"

namespace trrom
{

class Data;
class PDE_Constraint;
class InequalityOperators;
class ReducedObjectiveOperators;

template<typename ScalarType>
class Vector;

class AugmentedLagrangianTypeNP : public trrom::AugmentedLagrangianAssemblyMng
{
public:
    AugmentedLagrangianTypeNP(const std::tr1::shared_ptr<trrom::Data> & input_,
                              const std::tr1::shared_ptr<trrom::PDE_Constraint> & pde_,
                              const std::tr1::shared_ptr<trrom::ReducedObjectiveOperators> & objective_,
                              const std::vector<std::tr1::shared_ptr<trrom::InequalityOperators> > & inequality_);
    virtual ~AugmentedLagrangianTypeNP();

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
                 const std::tr1::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                 const double & tolerance_,
                 bool & inexactness_violated_);
    bool updateLagrangeMultipliers();

    int getEqualityEvaluationCounter() const;
    void updateEqualityEvaluationCounter();
    int getInverseJacobianStateCounter() const;
    void updateInverseJacobianStateCounter();
    int getAdjointInverseJacobianStateCounter() const;
    void updateInverseAdjointJacobianStateCounter();

    void setMinPenalty(double input_);
    void setPenaltyScaling(double input_);
    void useGaussNewtonHessian();

private:
    void computeObjectiveGradient(const trrom::Vector<double> & control_);
    void computeInequalityConstraintGradient(const trrom::Vector<double> & control_, trrom::Vector<double> & gradient_);
    void computeObjectiveHessianDual(const trrom::Vector<double> & control_, const trrom::Vector<double> & vector_);
    void computeObjectiveHessianTimesVector(const trrom::Vector<double> & control_,
                                            const trrom::Vector<double> & vector_,
                                            trrom::Vector<double> & hess_times_vector_);
    void computeInequalityConstraintHessianDual(const int & index_,
                                                const trrom::Vector<double> & control_,
                                                const trrom::Vector<double> & vector_);
    void computeInequalityHessianTimesVector(const trrom::Vector<double> & control_,
                                             const trrom::Vector<double> & vector_,
                                             trrom::Vector<double> & hess_times_vector_);
    void computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                  const trrom::Vector<double> & vector_,
                                  trrom::Vector<double> & hess_times_vec_);
    void computeGaussNewtonHessian(const trrom::Vector<double> & control_,
                                   const trrom::Vector<double> & vector_,
                                   trrom::Vector<double> & hess_times_vec_);

private:
    bool m_FullNewton;

    int m_HessianCounter;
    int m_GradientCounter;
    int m_ObjectiveCounter;
    int m_InequalityCounter;
    int m_InequalityGradientCounter;
    int m_EqualityEvaluationCounter;
    int m_InverseJacobianStateCounter;
    int m_AdjointInverseJacobianStateCounter;

    double m_Penalty;
    double m_MinPenalty;
    double m_PenaltyScaling;
    double m_NormLagrangianGradient;
    double m_NormInequalityConstraints;

    std::tr1::shared_ptr<trrom::Vector<double> > m_State;
    std::tr1::shared_ptr<trrom::Vector<double> > m_Slacks;
    std::tr1::shared_ptr<trrom::Vector<double> > m_DeltaState;
    std::tr1::shared_ptr<trrom::Vector<double> > m_HessWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_StateWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ObjectiveDual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ControlWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_GradientWorkVec;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LagrangianGradient;
    std::tr1::shared_ptr<trrom::Vector<double> > m_ObjectiveDeltaDual;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LagrangeMultipliers;
    std::tr1::shared_ptr<trrom::Vector<double> > m_InequalityConstraintValues;
    std::tr1::shared_ptr<trrom::Vector<double> > m_CurrentInequalityConstraintValues;

    std::vector<std::tr1::shared_ptr<trrom::Vector<double> > > m_InequalityDual;
    std::vector<std::tr1::shared_ptr<trrom::Vector<double> > > m_InequalityDeltaDual;

    std::tr1::shared_ptr<trrom::PDE_Constraint> m_PDE;
    std::tr1::shared_ptr<trrom::ReducedObjectiveOperators> m_Objective;
    std::vector<std::tr1::shared_ptr<trrom::InequalityOperators> > m_Inequality;

private:
    AugmentedLagrangianTypeNP(const trrom::AugmentedLagrangianTypeNP &);
    trrom::AugmentedLagrangianTypeNP & operator=(const trrom::AugmentedLagrangianTypeNP &);
};

}

#endif /* TRROM_AUGMENTEDLAGRANGIANTYPENP_HPP_ */
