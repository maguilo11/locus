/*
 * DOTk_Types.hpp
 *
 *  Created on: Aug 20, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TYPES_HPP_
#define DOTK_TYPES_HPP_

namespace dotk
{

namespace types
{
    enum functor_t
    {
        GRADIENT_TYPE_ULP = 1,
        GRADIENT_TYPE_UNP = 2,
    };

    enum problem_t
    {
        TYPE_ULP = 1,
        TYPE_UNLP = 2,
        TYPE_ELP = 3,
        TYPE_ENLP = 4,
        TYPE_LP_BOUND = 5,
        TYPE_NLP_BOUND = 6,
        TYPE_ELP_BOUND = 7,
        TYPE_ENLP_BOUND = 8,
        TYPE_CLP = 9,
        TYPE_CNLP = 10,
        TYPE_ILP = 11,
        PROBLEM_TYPE_UNDEFINED = 12,
    };

    enum numerical_integration_t
    {
        FORWARD_FINITE_DIFF = 1,
        BACKWARD_FINITE_DIFF = 2,
        CENTRAL_FINITE_DIFF = 3,
        SECOND_ORDER_FORWARD_FINITE_DIFF = 4,
        THIRD_ORDER_FORWARD_FINITE_DIFF = 5,
        THIRD_ORDER_BACKWARD_FINITE_DIFF = 6,
        NUM_INTG_DISABLED = 7,
    };

    enum linear_operator_t
    {
        HESSIAN_MATRIX = 1,
        IDENTITY_MATRIX = 2,
        AUGMENTED_SYSTEM = 3,
        USER_DEFINED_MATRIX = 4
    };

    enum projection_t
    {
        GRAM_SCHMIDT = 1,
        MODIFIED_GRAM_SCHMIDT = 2,
        HOUSEHOLDER = 3,
        ARNOLDI = 4,
        ARNOLDI_MODIFIED_GRAM_SCHMIDT = 5,
        ARNOLDI_HOUSEHOLDER = 6,
        INCOMPLETE_ORTHOGONALIZATION = 7,
        DIRECT_IMCOMPLETE_ORTHOGONALIZATION = 8,
        PROJECTION_DISABLED = 9
    };

    enum krylov_solver_t
    {
        LEFT_PREC_CG = 1,
        PREC_GMRES = 2,
        LEFT_PREC_CGNR = 3,
        LEFT_PREC_CGNE = 4,
        LEFT_PREC_CR = 5,
        LEFT_PREC_GCR = 6,
        PROJECTED_PREC_CG = 7,
        LANCZOS = 8,
        BICG = 9,
        BICG_STAB = 10,
        USER_DEFINED_KRYLOV_SOLVER = 11,
        KRYLOV_SOLVER_DISABLED = 12
    };

    enum direct_solver_t
    {
        DIRECT_SOLVER_DISABLED = 1,
        LOWER_TRIANGULAR_DIRECT_SOLVER = 2,
        UPPER_TRIANGULAR_DIRECT_SOLVER = 3,
        USER_DEFINED_DIRECT_SOLVER = 4
    };

    enum nonlinearcg_t
    {
        FLETCHER_REEVES_NLCG = 1,
        POLAK_RIBIERE_NLCG = 2,
        HESTENES_STIEFEL_NLCG = 3,
        CONJUGATE_DESCENT_NLCG = 4,
        HAGER_ZHANG_NLCG = 5,
        DAI_LIAO_NLCG = 6,
        DAI_YUAN_NLCG = 7,
        DAI_YUAN_HYBRID_NLCG = 8,
        PERRY_SHANNO_NLCG = 9,
        LIU_STOREY_NLCG = 10,
        DANIELS_NLCG = 11,
        UNDEFINED_NLCG = 12
    };

    enum line_search_t
    {
        BACKTRACKING_ARMIJO = 1,
        BACKTRACKING_GOLDSTEIN = 2,
        BACKTRACKING_CUBIC_INTRP = 3,
        LINE_SEARCH_HAGER_ZHANG = 4,
        GOLDENSECTION = 5,
        LINE_SEARCH_DISABLED = 6
    };

    enum gradient_t
    {
        FORWARD_DIFF_GRAD = 1,
        BACKWARD_DIFF_GRAD = 2,
        CENTRAL_DIFF_GRAD = 3,
        USER_DEFINED_GRAD = 4,
        PARALLEL_FORWARD_DIFF_GRAD = 5,
        PARALLEL_BACKWARD_DIFF_GRAD = 6,
        PARALLEL_CENTRAL_DIFF_GRAD = 7,
        GRADIENT_OPERATOR_DISABLED = 8
    };

    enum hessian_t
    {
        HESSIAN_DISABLED = 1,
        LBFGS_HESS = 2,
        LDFP_HESS = 3,
        LSR1_HESS = 4,
        SR1_HESS = 5,
        DFP_HESS = 6,
        USER_DEFINED_HESS = 7,
        USER_DEFINED_HESS_TYPE_CNP = 8,
        BARZILAIBORWEIN_HESS = 9,
    };

    enum invhessian_t
    {
        INV_HESS_DISABLED = 1,
        LBFGS_INV_HESS = 2,
        LDFP_INV_HESS = 3,
        LSR1_INV_HESS = 4,
        SR1_INV_HESS = 5,
        BFGS_INV_HESS = 6,
        USER_DEFINED_INV_HESS = 7,
        BARZILAIBORWEIN_INV_HESS = 8
    };

    enum trustregion_t
    {
        TRUST_REGION_CAUCHY = 1,
        TRUST_REGION_DOGLEG = 2,
        TRUST_REGION_DOUBLE_DOGLEG = 3,
        TRUST_REGION_DISABLED = 4
    };

    enum stop_criterion_t
    {
        NaN_TRIAL_STEP_NORM = 1,
        NaN_GRADIENT_NORM = 2,
        GRADIENT_TOL_SATISFIED = 3,
        TRIAL_STEP_TOL_SATISFIED = 4,
        OBJECTIVE_FUNC_TOL_SATISFIED = 5,
        MAX_NUM_ITR_REACHED = 6,
        OPTIMALITY_AND_FEASIBILITY_SATISFIED = 7,
        TRUST_REGION_RADIUS_SMALLER_THAN_TRIAL_STEP_NORM = 8,
        ACTUAL_REDUCTION_TOL_SATISFIED = 9,
        OPT_ALG_HAS_NOT_CONVERGED = 10
    };

    enum solver_stop_criterion_t
    {
        NaN_CURVATURE_DETECTED = 1,
        ZERO_CURVATURE_DETECTED = 2,
        NEGATIVE_CURVATURE_DETECTED = 3,
        INF_CURVATURE_DETECTED = 4,
        SOLVER_TOLERANCE_SATISFIED = 5,
        TRUST_REGION_VIOLATED = 6,
        MAX_SOLVER_ITR_REACHED = 7,
        SOLVER_DID_NOT_CONVERGED = 8,
        NaN_RESIDUAL_NORM = 9,
        INF_RESIDUAL_NORM = 10,
        INVALID_INEXACTNESS_MEASURE = 11,
        INVALID_ORTHOGONALITY_MEASURE = 12,
    };

    enum derivative_t
    {
        U = 1,
        Z = 2,
        UU = 3,
        ZZ = 4,
        UZ = 5,
        ZU = 6,
        ZERO_ORDER_DERIVATIVE = 7
    };

    enum display_t
    {
        ITERATION = 1,
        FINAL = 2,
        OFF = 3,
        DETAILED = 4
    };

    enum right_prec_t
    {
        RIGHT_PRECONDITIONER_DISABLED = 1,
        USER_DEFINED_RIGHT_PRECONDITIONER = 2
    };

    enum left_prec_t
    {
        LEFT_PRECONDITIONER_DISABLED = 1,
        SECANT_LEFT_PRECONDITIONER = 2,
        USER_DEFINED_LEFT_PRECONDITIONER = 3,
        AUGMENTED_SYSTEM_LEFT_PRECONDITIONER = 4
    };

    enum extent_t
    {
        NEW = 1,
        OLD = 2
    };

    enum constraint_method_t
    {
        CONSTRAINT_METHOD_DISABLED = 1,
        FEASIBLE_DIR = 2,
        PROJECTION_ALONG_FEASIBLE_DIR = 3,
        PROJECTED_GRADIENT = 4,
    };

    enum variable_t
    {
        STATE = 1,
        CONTROL = 2,
        DUAL = 3,
        PRIMAL = 4,
        UNDEFINED_VARIABLE = 5
    };

    enum bound_step_t
    {
        ARMIJO_STEP = 1,
        TRUST_REGION_STEP = 2,
        MIN_REDUCTION_STEP = 3,
        CONSTANT_STEP = 4
    };

    enum bound_t
    {
        LOWER_BOUND = 1,
        UPPER_BOUND = 2,
    };

    enum algorithm_t
    {
        DOTk_ALGORITHM_DISABLED = 1,
        NONLINEAR_CG = 2,
        LINE_SEARCH_QUASI_NEWTON = 3,
        TRUST_REGION_QUASI_NEWTON = 4,
        LINE_SEARCH_INEXACT_NEWTON = 5,
        TRUST_REGION_INEXACT_NEWTON = 6,
        INEXACT_TRUST_REGION_SQP = 7
    };

    enum stopping_criterion_t
    {
        FIX_CRITERION = 1,
        RELATIVE_CRITERION = 2,
        SQP_DUAL_PROBLEM_CRITERION = 3,
        TANGENTIAL_PROBLEM_CRITERION = 4,
        QUASI_NORMAL_PROBLEM_CRITERION = 5,
        TANGENTIAL_SUBPROBLEM_CRITERION = 6,
    };

    enum stopping_criterion_param_t
    {
        NORM_GRADIENT = 1,
        NORM_RESIDUAL = 2,
        FIX_TOLERANCE = 3,
        DUAL_TOLERANCE = 4,
        RELATIVE_TOLERANCE = 5,
        TRUST_REGION_RADIUS = 6,
        TANGENTIAL_TOLERANCE = 7,
        DUAL_DOT_GRAD_TOLERANCE = 8,
        QUASI_NORMAL_STOPPING_TOL = 9,
        CURRENT_KRYLOV_SOLVER_ITR = 10,
        TRUST_REGION_RADIUS_PENALTY = 11,
        PROJECTED_GRADIENT_TOLERANCE = 12,
        NORM_TANGENTIAL_STEP_RESIDUAL = 13,
        NORM_PROJECTED_TANGENTIAL_STEP = 14,
        TANGENTIAL_TOL_CONTRACTION_FACTOR = 15,
    };

    enum qr_t
    {
        UNDEFINED_QR_METHOD = 1,
        CLASSICAL_GRAM_SCHMIDT_QR = 2,
        MODIFIED_GRAM_SCHMIDT_QR = 3,
        HOUSEHOLDER_QR = 4,
        USER_DEFINED_QR_METHOD = 5,
    };

    enum eigen_t
    {
        UNDEFINED_EIGEN_METHOD = 1,
        QR_EIGEN_METHOD = 2,
        POWER_METHOD = 3,
        RAYLEIGH_QUOTIENT_METHOD = 4,
        RAYLEIGH_RITZ_METHOD = 5,
        USER_DEFINED_EIGEN_METHOD = 6,
    };
};

}

typedef int Int;
typedef double Real;

#endif /* DOTK_TYPES_HPP_ */
