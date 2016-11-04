# Set DOTk PATH
DOTk_INSTALL_DIR = /Users/miguelaguilo/dotk/

# Google test directory 
GTEST_INCLUDE_DIR = /Users/miguelaguilo/Software/gtest/include/
GTEST_SRC = /Users/miguelaguilo/Software/gtest_mybuild_clang++/CMakeFiles/gtest.dir/src/

# Unit tests directory
UNITDIR = /Users/miguelaguilo/dotk/unittest/

# Set compiler options
#CXX = /usr/bin/clang++
#CXXFLAGS = -g -std=c++11 -stdlib=libstdc++
CXX = /usr/local/bin/mpicxx
CXXFLAGS =-fopenmp -O3 -g -std=c++11

# C++ include directory
#CXX_INCLUDE = /usr/include/c++/4.2.1/
CXX_INCLUDE = /usr/local/Cellar/gcc/4.9.2_1/include/c++/4.9.2/

# Set DOTk source and include directories
DOTk_SOURCE_DIR = $(DOTk_INSTALL_DIR)src/
DOTk_INCLUDE_DIR = $(DOTk_INSTALL_DIR)include/

# Set DOTk include directories
DOTk_EIGEN=$(DOTk_INCLUDE_DIR)DOTk_Eigen/
DOTk_VECTOR=$(DOTk_INCLUDE_DIR)DOTk_Vector/
DOTk_MATRIX=$(DOTk_INCLUDE_DIR)DOTk_Matrix/
DOTk_CCSA=$(DOTk_INCLUDE_DIR)DOTk_MethodCCSA/
DOTk_FUNCTOR=$(DOTk_INCLUDE_DIR)DOTk_Functor/
DOTk_HESSIAN=$(DOTk_INCLUDE_DIR)DOTk_Hessian/
DOTk_FACTORY=$(DOTk_INCLUDE_DIR)DOTk_Factory/
DOTk_IO_TOOLS=$(DOTk_INCLUDE_DIR)DOTk_IOTools/
DOTk_VARIABLE=$(DOTk_INCLUDE_DIR)DOTk_Variable/
DOTk_GRADIENT=$(DOTk_INCLUDE_DIR)DOTk_Gradient/
DOTk_OC=$(DOTk_INCLUDE_DIR)DOTk_OptimalityCriteria/
DOTk_QR=$(DOTk_INCLUDE_DIR)DOTk_OrthogonalFactorization/
DOTk_NUM_INTG=$(DOTk_INCLUDE_DIR)DOTk_NumericalDifferentiation/
DOTk_INV_HESSIAN=$(DOTk_INCLUDE_DIR)DOTk_InvHessian/
DOTk_LINE_SEARCH=$(DOTk_INCLUDE_DIR)DOTk_LineSearch/
DOTk_TRUST_REGION=$(DOTk_INCLUDE_DIR)DOTk_TrustRegion/
DOTk_NONLINEAR_CG=$(DOTk_INCLUDE_DIR)DOTk_NonlinearCG/
DOTk_BOUND_CONSTRAINT=$(DOTk_INCLUDE_DIR)DOTk_BoundConstraint/
DOTk_LINE_SEARCH_ALG_DATA_MNG=$(DOTk_INCLUDE_DIR)DOTk_LineSearchAlgorithmsDataMng/
DOTk_TRUST_REGION_ALG_DATA_MNG=$(DOTk_INCLUDE_DIR)DOTk_TrustRegionAlgorithmsDataMng/
DOTk_FIRST_ORDER_ALG=$(DOTk_INCLUDE_DIR)DOTk_FirstOrderAlgorithm/
DOTk_GRAD_ALG_TOOLS=$(DOTk_INCLUDE_DIR)DOTk_GradAlgorithmsTools/
DOTk_LINEAR_OPERATOR=$(DOTk_INCLUDE_DIR)DOTk_LinearOperator/
DOTk_DIRECT_SOLVERS=$(DOTk_INCLUDE_DIR)DOTk_DirectSolvers/
DOTk_LEFT_PREC=$(DOTk_INCLUDE_DIR)DOTk_LeftPreconditioners/
DOTk_RIGHT_PREC=$(DOTk_INCLUDE_DIR)DOTk_RightPreconditioners/
DOTk_ORTHO_PROJ=$(DOTk_INCLUDE_DIR)DOTk_OrthogonalProjections/
DOTk_KRYLOV_SOLVER=$(DOTk_INCLUDE_DIR)DOTk_KrylovSolvers/
DOTk_KRYLOV_SOLVER_MNG=$(DOTk_INCLUDE_DIR)DOTk_KrylovSolverDataMng/
DOTk_KRYLOV_SOLVER_TOLERANCE_CRITERION=$(DOTk_INCLUDE_DIR)DOTk_KrylovSolverStoppingCriterion/
DOTk_INEXACT_NEWTON_ALG=$(DOTk_INCLUDE_DIR)DOTk_InexactNewtonAlgorithms/
DOTk_INEXACT_NEWTON_ALG_BOUND=$(DOTk_INCLUDE_DIR)DOTk_InexactNewtonAlgorithmsBound/
DOTk_DIAGNOSTICS_TOOLS=$(DOTk_INCLUDE_DIR)DOTk_DiagnosticsTools/
DOTk_SQP_ALG=$(DOTk_INCLUDE_DIR)DOTk_SqpAlgorithm/
DOTk_OBJECTIVE_FUNC_OP=$(DOTk_INCLUDE_DIR)DOTk_ObjectiveFunction/
DOTk_EQ_CONSTRAINT_OP=$(DOTk_INCLUDE_DIR)DOTk_EqualityConstraint/
DOTk_INEQ_CONSTRAINT_OP=$(DOTk_INCLUDE_DIR)DOTk_InequalityConstraint/
DOTk_OPERATORS_ROUTINES_MNG=$(DOTk_INCLUDE_DIR)DOTk_AssemblyManager/
DOTk_STEIHAUG_TOINT=$(DOTk_INCLUDE_DIR)DOTk_SteihaugToint/

OBJS = $(DOTk_SOURCE_DIR)DOTk_LineSearchAlgorithmsDataMng/DOTk_OptimizationDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_Variable/DOTk_Variable.o \
	$(DOTk_SOURCE_DIR)DOTk_Variable/DOTk_Dual.o \
	$(DOTk_SOURCE_DIR)DOTk_Variable/DOTk_State.o \
	$(DOTk_SOURCE_DIR)DOTk_Variable/DOTk_Primal.o \
	$(DOTk_SOURCE_DIR)DOTk_Variable/DOTk_Control.o \
	$(DOTk_SOURCE_DIR)DOTk_Functor/DOTk_Functor.o \
	$(DOTk_SOURCE_DIR)DOTk_Functor/DOTk_GradientTypeULP.o \
	$(DOTk_SOURCE_DIR)DOTk_Functor/DOTk_GradientTypeUNP.o \
	$(DOTk_SOURCE_DIR)DOTk_Eigen/DOTk_EigenQR.o \
	$(DOTk_SOURCE_DIR)DOTk_Eigen/DOTk_EigenUtils.o \
	$(DOTk_SOURCE_DIR)DOTk_Eigen/DOTk_EigenMethod.o \
	$(DOTk_SOURCE_DIR)DOTk_Eigen/DOTk_PowerMethod.o \
	$(DOTk_SOURCE_DIR)DOTk_Eigen/DOTk_RayleighRitz.o \
	$(DOTk_SOURCE_DIR)DOTk_Eigen/DOTk_RayleighQuotient.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_OmpArray.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_OmpVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_MpiArray.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_MpiVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_MpiX_Array.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_MpiX_Vector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_SerialArray.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_SerialVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_ParallelUtils.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_MultiVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_PrimalVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Matrix/DOTk_RowMatrix.o \
	$(DOTk_SOURCE_DIR)DOTk_Matrix/DOTk_DenseMatrix.o \
	$(DOTk_SOURCE_DIR)DOTk_Matrix/DOTk_ColumnMatrix.o \
	$(DOTk_SOURCE_DIR)DOTk_Matrix/DOTk_UpperTriangularMatrix.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointPcg.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_KelleySachsStepMng.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointSolver.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointNewton.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointLinMore.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointStepMng.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointNewtonIO.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointKelleySachs.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointProjGradStep.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_ProjectedSteihaugTointPcg.o \
	$(DOTk_SOURCE_DIR)DOTk_OptimalityCriteria/DOTk_OptimalityCriteria.o \
	$(DOTk_SOURCE_DIR)DOTk_OptimalityCriteria/DOTk_OptimalityCriteriaDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_OptimalityCriteria/DOTk_OptimalityCriteriaRoutineMng.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_UtilsCCSA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_DataMngCCSA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTK_MethodCcsaIO.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_AlgorithmCCSA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_SubProblemMMA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_DualSolverCCSA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_DualSolverNLCG.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_SubProblemCCSA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_SubProblemGCMMA.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_DataMngNonlinearCG.o \
	$(DOTk_SOURCE_DIR)DOTk_MethodCCSA/DOTk_DualObjectiveFunctionMMA.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_AssemblyManager.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_RoutinesTypeLP.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_RoutinesTypeNP.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_RoutinesTypeULP.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_RoutinesTypeUNP.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_RoutinesTypeENP.o \
	$(DOTk_SOURCE_DIR)DOTk_AssemblyManager/DOTk_RoutinesTypeELP.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalFactorization/DOTk_QR.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalFactorization/DOTk_Householder.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalFactorization/DOTk_ModifiedGramSchmidt.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalFactorization/DOTk_ClassicalGramSchmidt.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalFactorization/DOTk_OrthogonalFactorization.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_Rosenbrock.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_BealeObjective.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_PowellObjective.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_ZakharovObjective.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_ObjectiveFunctionMmaTest.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_FreudensteinRothObjective.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_NocedalAndWrightObjective.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_GcmmaTestObjectiveFunction.o \
	$(DOTk_SOURCE_DIR)DOTk_EqualityConstraint/DOTk_NocedalAndWrightEquality.o \
	$(DOTk_SOURCE_DIR)DOTk_ObjectiveFunction/DOTk_NocedalAndWrightObjectiveNLP.o \
	$(DOTk_SOURCE_DIR)DOTk_EqualityConstraint/DOTk_NocedalAndWrightEqualityNLP.o \
	$(DOTk_SOURCE_DIR)DOTk_InequalityConstraint/DOTk_GcmmaTestInequalityConstraint.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_BackwardFiniteDifference.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_CentralFiniteDifference.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_ForwardFiniteDifference.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_NumericalDifferentiation.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_SecondOrderForwardFiniteDifference.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_ThirdOrderBackwardFiniteDifference.o \
	$(DOTk_SOURCE_DIR)DOTk_NumericalDifferentiation/DOTk_ThirdOrderForwardFiniteDifference.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearchAlgorithmsDataMng/DOTk_LineSearchAlgorithmsDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearchAlgorithmsDataMng/DOTk_LineSearchMngTypeULP.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearchAlgorithmsDataMng/DOTk_LineSearchMngTypeUNP.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegionAlgorithmsDataMng/DOTk_TrustRegionAlgorithmsDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegionAlgorithmsDataMng/DOTk_TrustRegionMngTypeULP.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegionAlgorithmsDataMng/DOTk_TrustRegionMngTypeUNP.o \
	$(DOTk_SOURCE_DIR)DOTk_GradAlgorithmsTools/DOTk_MathUtils.o \
	$(DOTk_SOURCE_DIR)DOTk_GradAlgorithmsTools/DOTk_VariablesUtils.o \
	$(DOTk_SOURCE_DIR)DOTk_GradAlgorithmsTools/DOTk_GradBasedIoUtils.o \
	$(DOTk_SOURCE_DIR)DOTk_GradAlgorithmsTools/DOTk_DescentDirectionTools.o \
	$(DOTk_SOURCE_DIR)DOTk_GradAlgorithmsTools/DOTk_NonLinearProgrammingUtils.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_ConjugateDescent.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_Daniels.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_DaiLiao.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_DaiYuan.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_DaiYuanHybrid.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_DescentDirection.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_HagerZhang.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_HestenesStiefel.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_FletcherReeves.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_LiuStorey.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_PerryShanno.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_PolakRibiere.o \
	$(DOTk_SOURCE_DIR)DOTk_NonlinearCG/DOTk_ScaleParametersNLCG.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_UserDefinedGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_BackwardDifferenceGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_CentralDifferenceGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_FirstOrderOperator.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_ForwardDifferenceGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_ParallelBackwardDiffGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_ParallelCentralDiffGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Gradient/DOTk_ParallelForwardDiffGrad.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_UserDefinedHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_BarzilaiBorweinHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_DFPHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_LBFGSHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_LDFPHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_LSR1Hessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_SecondOrderOperator.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_SR1Hessian.o \
	$(DOTk_SOURCE_DIR)DOTk_Hessian/DOTk_UserDefinedHessianTypeCNP.o \
	$(DOTk_SOURCE_DIR)DOTk_InvHessian/DOTk_BarzilaiBorweinInvHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_InvHessian/DOTk_BFGSInvHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_InvHessian/DOTk_LDFPInvHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_InvHessian/DOTk_LBFGSInvHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_InvHessian/DOTk_LSR1InvHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_InvHessian/DOTk_SR1InvHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_BoundConstraint/DOTk_BoundConstraint.o \
	$(DOTk_SOURCE_DIR)DOTk_BoundConstraint/DOTk_BoundConstraints.o \
	$(DOTk_SOURCE_DIR)DOTk_BoundConstraint/DOTk_FeasibleDirection.o \
	$(DOTk_SOURCE_DIR)DOTk_BoundConstraint/DOTk_GradientProjectionMethod.o \
	$(DOTk_SOURCE_DIR)DOTk_BoundConstraint/DOTk_ProjectionAlongFeasibleDir.o \
	$(DOTk_SOURCE_DIR)DOTk_IOTools/DOTk_InexactTrustRegionSqpIO.o \
	$(DOTk_SOURCE_DIR)DOTk_IOTools/DOTk_FirstOrderLineSearchAlgIO.o \
	$(DOTk_SOURCE_DIR)DOTk_IOTools/DOTk_LineSearchInexactNewtonIO.o \
	$(DOTk_SOURCE_DIR)DOTk_IOTools/DOTk_TrustRegionInexactNewtonIO.o \
	$(DOTk_SOURCE_DIR)DOTk_FirstOrderAlgorithm/DOTk_FirstOrderAlgorithm.o \
	$(DOTk_SOURCE_DIR)DOTk_FirstOrderAlgorithm/DOTk_NonlinearCG.o \
	$(DOTk_SOURCE_DIR)DOTk_FirstOrderAlgorithm/DOTk_LineSearchQuasiNewton.o \
	$(DOTk_SOURCE_DIR)DOTk_InexactNewtonAlgorithms/DOTk_InexactNewtonAlgorithms.o \
	$(DOTk_SOURCE_DIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewton.o \
	$(DOTk_SOURCE_DIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewton.o \
	$(DOTk_SOURCE_DIR)DOTk_SqpAlgorithm/DOTk_TrustRegionMngTypeELP.o \
	$(DOTk_SOURCE_DIR)DOTk_SqpAlgorithm/DOTk_InexactTrustRegionSQP.o \
	$(DOTk_SOURCE_DIR)DOTk_SqpAlgorithm/DOTk_SequentialQuadraticProgramming.o \
	$(DOTk_SOURCE_DIR)DOTk_SqpAlgorithm/DOTk_InexactTrustRegionSqpSolverMng.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_BoundConstraintFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_DirectSolverFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_FirstOrderOperatorFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_HessianFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_InverseHessianFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_KrylovSolverFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_LineSearchFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_AugmentedSystemPrecFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_SecantLeftPreconditionerFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_NonlinearCGFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_NumericalDifferentiatonFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_OrthogonalProjectionFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_RightPreconditionerFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_Factory/DOTk_TrustRegionFactory.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_LineSearch.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_ProjectedStep.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_ArmijoLineSearch.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_BacktrackingCubicInterpolation.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_GoldenSectionLineSearch.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_GoldsteinLineSearch.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_HagerZhangLineSearch.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_LineSearchStep.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearch/DOTk_ProjectedLineSearchStep.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegion/DOTk_TrustRegion.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegion/DOTk_DoglegTrustRegion.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegion/DOTk_TrustRegionStepMng.o \
	$(DOTk_SOURCE_DIR)DOTk_TrustRegion/DOTk_DoubleDoglegTrustRegion.o \
	$(DOTk_SOURCE_DIR)DOTk_DirectSolvers/DOTk_DirectSolver.o \
	$(DOTk_SOURCE_DIR)DOTk_DirectSolvers/DOTk_UpperTriangularDirectSolver.o \
	$(DOTk_SOURCE_DIR)DOTk_DirectSolvers/DOTk_LowerTriangularDirectSolver.o \
	$(DOTk_SOURCE_DIR)DOTk_LeftPreconditioners/DOTk_Preconditioner.o \
	$(DOTk_SOURCE_DIR)DOTk_LeftPreconditioners/DOTk_LeftPreconditioner.o \
	$(DOTk_SOURCE_DIR)DOTk_LeftPreconditioners/DOTk_AugmentedSystemLeftPrec.o \
	$(DOTk_SOURCE_DIR)DOTk_LeftPreconditioners/DOTK_SecantLeftPreconditioner.o \
	$(DOTk_SOURCE_DIR)DOTk_RightPreconditioners/DOTk_RightPreconditioner.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalProjections/DOTk_GramSchmidt.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalProjections/DOTk_ArnoldiProjection.o \
	$(DOTk_SOURCE_DIR)DOTk_OrthogonalProjections/DOTk_OrthogonalProjection.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_FixedCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_QuasiNormalProbCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_RelativeCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_SqpDualProblemCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_TangentialProblemCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_TangentialSubProblemCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverStoppingCriterion/DOTk_KrylovSolverStoppingCriterion.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_KrylovSolverDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_PrecGenMinResDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_ProjLeftPrecCgDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_LeftPrecConjGradDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_LeftPrecCGNResDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_LeftPrecCGNEqDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_LeftPrecConjResDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolverDataMng/DOTk_LeftPrecGenConjResDataMng.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_KrylovSolver.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_LeftPrecCG.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_LeftPrecCGNR.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_LeftPrecCGNE.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_LeftPrecCR.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_LeftPrecGCR.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_PrecGMRES.o \
	$(DOTk_SOURCE_DIR)DOTk_KrylovSolvers/DOTk_ProjectedLeftPrecCG.o \
	$(DOTk_SOURCE_DIR)DOTk_LinearOperator/DOTk_Hessian.o \
	$(DOTk_SOURCE_DIR)DOTk_LinearOperator/DOTk_LinearOperator.o \
	$(DOTk_SOURCE_DIR)DOTk_LinearOperator/DOTk_AugmentedSystem.o \
	$(DOTk_SOURCE_DIR)DOTk_LinearOperator/DOTk_GmresTestMatrix.o \
	$(DOTk_SOURCE_DIR)DOTk_LinearOperator/DOTk_NumericallyDifferentiatedHessian.o \
	$(DOTk_SOURCE_DIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsTypeLP.o \
	$(DOTk_SOURCE_DIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsEqualityTypeNP.o \
	$(DOTk_SOURCE_DIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsObjectiveTypeNP.o \
	$(DOTk_SOURCE_DIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsInequalityTypeNP.o \
	$(DOTk_SOURCE_DIR)DOTk_DiagnosticsTools/DOTk_DerivativeDiagnosticsTool.o

UNIT = $(GTEST_SRC)gtest-all.cc.o \
	$(UNITDIR)UnitMain.o \
	$(UNITDIR)DOTk_Eigen/DOTk_EigenTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_MpiArrayTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_MpiVectorTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_OmpArrayTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_OmpVectorTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_MpiX_ArrayTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_MpiX_VectorTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_SerialArrayTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_SerialVectorTest.o \
	$(UNITDIR)DOTk_Vector/DOTk_MultiVectorTest.o \
	$(UNITDIR)DOTk_SteihaugToint/DOTk_SteihaugTointTest.o \
	$(UNITDIR)DOTk_SteihaugToint/DOTk_KelleySachsTrustRegionTest.o \
	$(UNITDIR)DOTk_SteihaugToint/DOTk_SteihaugTointProjGradStepTest.o \
	$(UNITDIR)DOTk_Matrix/DOTk_RowMatrixTest.o \
	$(UNITDIR)DOTk_Matrix/DOTk_ColumnMatrixTest.o \
	$(UNITDIR)DOTk_Matrix/DOTk_SerialDenseMatrixTest.o \
	$(UNITDIR)DOTk_Matrix/DOTk_UpperTriangularMatrixTest.o \
	$(UNITDIR)DOTk_Variable/DOTk_VariableTest.o \
	$(UNITDIR)DOTk_GtestTools/DOTk_GtestDOTkVecTools.o \
	$(UNITDIR)DOTk_OptimalityCriteria/DOTk_OptimalityCriteriaTest.o \
	$(UNITDIR)DOTk_MethodCCSA/DOTk_MethodCCSA_Test.o \
	$(UNITDIR)DOTk_NumericalDifferentiation/DOTk_NumericalDifferentiationTest.o \
	$(UNITDIR)DOTk_LineSearchAlgorithmsDataMng/DOTk_LineSearchMngTypeULPTest.o \
	$(UNITDIR)DOTk_BoundConstraint/DOTk_BoundConstraintTest.o \
	$(UNITDIR)DOTk_BoundConstraint/DOTk_FeasibleDirectionTest.o \
	$(UNITDIR)DOTk_BoundConstraint/DOTk_GradientProjectionMethodTest.o \
	$(UNITDIR)DOTk_BoundConstraint/DOTk_ProjectionAlongFeasibleDirTest.o \
	$(UNITDIR)DOTk_LineSearch/DOTk_ArmijoLineSearchTest.o \
	$(UNITDIR)DOTk_LineSearch/DOTk_BacktrackingCubicInterpolationTest.o \
	$(UNITDIR)DOTk_LineSearch/DOTk_HagerZhangLineSearchTest.o \
	$(UNITDIR)DOTk_LineSearch/DOTk_GoldenSectionLineSearchTest.o \
	$(UNITDIR)DOTk_LineSearch/DOTk_GoldsteinLineSearchTest.o \
	$(UNITDIR)DOTk_TrustRegion/DOTk_TrustRegionTest.o \
	$(UNITDIR)DOTk_TrustRegion/DOTk_DoglegTrustRegionTest.o \
	$(UNITDIR)DOTk_TrustRegion/DOTk_DoubleDoglegTrustRegionTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_FletcherReevesTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_PolakRibiereTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_HestenesStiefelTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_ConjugateDescentTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_DaiYuanTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_DaiYuanHybridTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_HagerZhangTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_DaiLiaoTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_PerryShannoTest.o \
	$(UNITDIR)DOTk_NonlinearCG/DOTk_LiuStoreyTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_SecondOrderOperatorTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_BarzilaiBorweinHessianTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_LBFGSHessianTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_LDFPHessianTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_LSR1HessianTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_DFPHessianTest.o \
	$(UNITDIR)DOTk_Hessian/DOTk_SR1HessianTest.o \
	$(UNITDIR)DOTk_InvHessian/DOTk_LBFGSInvHessianTest.o \
	$(UNITDIR)DOTk_InvHessian/DOTk_LDFPInvHessianTest.o \
	$(UNITDIR)DOTk_InvHessian/DOTk_LSR1InvHessianTest.o \
	$(UNITDIR)DOTk_InvHessian/DOTk_BFGSInvHessianTest.o \
	$(UNITDIR)DOTk_InvHessian/DOTk_SR1InvHessianTest.o \
	$(UNITDIR)DOTk_InvHessian/DOTk_BarzilaiBorweinInvHessianTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_FirstOrderOperatorTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_BackwardDifferenceGradTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_ForwardDifferenceGradTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_CentralDifferenceGradTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_ParallelCentralDiffGradTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_ParallelForwardDiffGradTest.o \
	$(UNITDIR)DOTk_Gradient/DOTk_ParallelBackwardDiffGradTest.o \
	$(UNITDIR)DOTk_FreeFunctions/DOTk_FreeFunctionsTest.o \
	$(UNITDIR)DOTk_OrthogonalProjections/DOTk_GramSchmidtTest.o \
	$(UNITDIR)DOTk_OrthogonalProjections/DOTk_ArnoldiProjectionTest.o \
	$(UNITDIR)DOTk_OrthogonalFactorization/DOTk_QrFreeFunctionsTest.o \
	$(UNITDIR)DOTk_OrthogonalFactorization/DOTk_OrthoFactorizationTest.o \
	$(UNITDIR)DOTk_DirectSolvers/DOTk_DirectSolverTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_PrecGMRESTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_LeftPrecCGTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_LeftPrecCgneTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_LeftPrecCgnrTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_LeftPrecGcrTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_LeftPrecCrTest.o \
	$(UNITDIR)DOTk_KrylovSolverStoppingCriterion/DOTk_StoppingCriterionTest.o \
	$(UNITDIR)DOTk_FirstOrderAlgorithm/DOTk_NonlinearCGTest.o \
	$(UNITDIR)DOTk_FirstOrderAlgorithm/DOTk_NonlinearCGBoundTest.o \
	$(UNITDIR)DOTk_FirstOrderAlgorithm/DOTk_FirstOrderAlgorithmTest.o \
	$(UNITDIR)DOTk_FirstOrderAlgorithm/DOTk_LineSearchQuasiNewtonTest.o \
	$(UNITDIR)DOTk_FirstOrderAlgorithm/DOTk_LineSearchQuasiNewtonBoundTest.o \
	$(UNITDIR)DOTk_FirstOrderAlgorithm/DOTk_FirstOrderAlgorithmsFreudensteinRothTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewtonPcgTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewtonGmresTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewtonCgnrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewtonCgneTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewtonCrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_LineSearchInexactNewtonGcrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewtonCgneTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewtonCgnrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewtonCrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewtonGcrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewtonGmresTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithms/DOTk_TrustRegionInexactNewtonPcgTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithmsBound/DOTk_BoundLineSearchInexactNewtonPcgTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithmsBound/DOTk_BoundLineSearchInexactNewtonCgneTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithmsBound/DOTk_BoundLineSearchInexactNewtonCrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithmsBound/DOTk_BoundLineSearchInexactNewtonGcrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithmsBound/DOTk_BoundLineSearchInexactNewtonGmresTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgorithmsBound/DOTk_BoundLineSearchInexactNewtonCgnrTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgBoundNumIntgHess/DOTk_BoundLineSearchIxNewtonCgneNumIntgHessTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgBoundNumIntgHess/DOTk_BoundLineSearchIxNewtonCgnrNumIntgHessTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgBoundNumIntgHess/DOTk_BoundLineSearchIxNewtonCrNumIntgHessTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgBoundNumIntgHess/DOTk_BoundLineSearchIxNewtonGcrNumIntgHessTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgBoundNumIntgHess/DOTk_BoundLineSearchIxNewtonGmresNumIntgHessTest.o \
	$(UNITDIR)DOTk_InexactNewtonAlgBoundNumIntgHess/DOTk_BoundLineSearchIxNewtonPcgNumIntgHessTest.o \
	$(UNITDIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsLP_Test.o \
	$(UNITDIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsTypeULPTest.o \
	$(UNITDIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsEqualityTypeNPTest.o \
	$(UNITDIR)DOTk_DiagnosticsTools/DOTk_DiagnosticsObjectiveTypeNPTest.o \
	$(UNITDIR)DOTk_KrylovSolvers/DOTk_ProjLeftPrecCgTest.o \
	$(UNITDIR)DOTk_SequentialQuadraticProgramming/DOTk_InexactTrustRegionSQPTest.o

.cpp.o:
	$(CXX) $(CXXFLAGS) -I$(CXX_INCLUDE) -I$(GTEST_INCLUDE_DIR) -Isrc/DOTk_Vector -Isrc/DOTk_Matrix -Iinclude \
	-I$(DOTk_VECTOR) -I$(DOTk_MATRIX) -I$(DOTk_HESSIAN) -I$(DOTk_FUNCTOR) -I$(DOTk_INEQ_CONSTRAINT_OP) \
	-I$(DOTk_LINE_SEARCH_ALG_DATA_MNG) -I$(DOTk_LINE_SEARCH) -I$(DOTk_HESSIAN) -I$(DOTk_GRADIENT) -I$(DOTk_NUM_INTG) \
	-I$(DOTk_INV_HESSIAN) -I$(DOTk_FACTORY) -I$(DOTk_TRUST_REGION) -I$(DOTk_NONLINEAR_CG) -I$(DOTk_VARIABLE) \
	-I$(DOTk_BOUND_CONSTRAINT) -I$(DOTk_IO_TOOLS) -I$(DOTk_FIRST_ORDER_ALG) -I$(DOTk_GRAD_ALG_TOOLS) -I$(DOTk_QR) \
	-I$(DOTk_LINEAR_OPERATOR) -I$(DOTk_DIRECT_SOLVERS) -I$(DOTk_LEFT_PREC) -I$(DOTk_RIGHT_PREC) -I$(DOTk_SQP_ALG) \
	-I$(DOTk_ORTHO_PROJ) -I$(DOTk_KRYLOV_SOLVER_MNG) -I$(DOTk_KRYLOV_SOLVER) -I$(DOTk_INEXACT_NEWTON_ALG) -I$(DOTk_OC) \
	-I$(DOTk_TRUST_REGION_ALG_DATA_MNG) -I$(DOTk_DIAGNOSTICS_TOOLS) -I$(DOTk_KRYLOV_SOLVER_TOLERANCE_CRITERION) \
	-I$(DOTk_EQ_CONSTRAINT_OP) -I$(DOTk_OPERATORS_ROUTINES_MNG) -I$(DOTk_OBJECTIVE_FUNC_OP) -Iunittest/include \
	-I$(DOTk_EIGEN) -I$(DOTk_CCSA) -I$(DOTk_STEIHAUG_TOINT) -c -o $@ $<

dotk: $(OBJS) $(DOTk_SOURCE_DIR)line_search_inexact_newton_main.o
	$(CXX) $(CXXFLAGS) -o dotk $^ 	

unit: $(OBJS) $(UNIT)
	$(CXX) $(CXXFLAGS) -o unit $^
	
matrix:  $(OBJS) $(MATRIX)
	$(CXX) $(CXXFLAGS) -o matrix $^

clean:
	find . -name "*.o" -exec rm {} \;
	rm -f dotk unit matrix
