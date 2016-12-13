# Set DOTk install directory
DOTK_INSTALL_DIR=/Users/miguelaguilo/locus/

# Set MEX compiler (MEX) and corresponding flags (MEXFLAGS)
MEXFLAGS=
MEX=/Applications/MATLAB_R2015a.app/bin/mex 
CXX=/usr/local/bin/mpicxx

# C++ include directory
CXX_HEADERS=/usr/local/Cellar/gcc/4.9.2_1/include/c++/4.9.2/
MEX_HEADERS=/Applications/MATLAB_R2015a.app/extern/include/

# Include directories
ALL_INCLUDE_DIR=-I$(CXX_HEADERS) -I$(MEX_HEADERS) -I$(MEX_INCLUDE_DIR) -I$(MEX_PARSER) -I$(MEX_ALG_TYPE_FO) \
	-I$(MEX_OPERATOR_API) -I$(MEX_ALG_TYPE_NEWTON) -I$(MEX_FACTORIES) -I$(MEX_ALG_TYPE_SQP) -I$(MEX_ALG_TYPE_CCSA) \
	-I$(MEX_ALG_TYPE_OC) -I$(MEX_UTILITIES)  -I$(MEX_FACTORIES_SRC) -I$(MEX_DIAGNOSTICS) -I$(MEX_OPERATOR_API_SRC) \
	-I$(MEX_PTR) -I$(DOTk_INV_HESSIAN) -I$(DOTk_FACTORY) -I$(DOTk_TRUST_REGION) -I$(DOTk_NONLINEAR_CG) -I$(DOTk_VARIABLE) \
	-I$(DOTk_BOUND_CONSTRAINT) -I$(DOTk_IO_TOOLS) -I$(DOTk_FIRST_ORDER_ALG) -I$(DOTk_GRAD_ALG_TOOLS) -I$(DOTk_HESSIAN) \
	-I$(DOTk_LINEAR_OPERATOR) -I$(DOTk_DIRECT_SOLVERS) -I$(DOTk_LEFT_PREC) -I$(DOTk_RIGHT_PREC) -I$(DOTk_SQP_ALG) \
	-I$(DOTk_ORTHO_PROJ) -I$(DOTk_KRYLOV_SOLVER_MNG) -I$(DOTk_KRYLOV_SOLVER) -I$(DOTk_INEXACT_NEWTON_ALG) -I$(DOTk_FUNCTOR) \
	-I$(DOTk_TRUST_REGION_ALG_DATA_MNG) -I$(DOTk_DIAGNOSTICS_TOOLS) -I$(DOTk_KRYLOV_SOLVER_TOLERANCE_CRITERION) \
	-I$(DOTk_EQ_CONSTRAINT_OP) -I$(DOTk_INEQ_CONSTRAINT_OP) -I$(DOTk_OPERATORS_ROUTINES_MNG) -I$(DOTk_EIGEN) \
	-I$(DOTk_NUM_INTG) -I$(DOTk_GRADIENT) -I$(DOTk_OBJECTIVE_FUNC_OP) -I$(DOTk_VECTOR) -I$(DOTk_CCSA) \
	-I$(DOTk_HESSIAN) -I$(DOTk_MATRIX) -I$(DOTk_LINE_SEARCH_ALG_DATA_MNG) -I$(DOTk_VECTOR_SRC) -I$(DOTk_LINE_SEARCH) \
	-I$(DOTk_MATRIX_SRC) -I$(DOTk_OC) -I$(DOTk_QR) -I$(DOTk_STEIHAUG_TOINT) -I$(DOTk_INCLUDE_DIR) 

# Set GNU compiler options
CXXOPTIMFLAGS='-O3'
CXXFLAGS='-std=c++11 -Wall -fopenmp'

# Set MEX source directory (MEX_SOURCE_DIR), MEX include directory (MEX_INCLUDE_DIR), objects output directory (MEX_OBJECT_DIR), 
# and executables output directory (EXEDIR)
MEX_SOURCE_DIR=src/
MEX_OBJECT_DIR=obj/
EXEDIR=exe/
MEX_INCLUDE_DIR=include/

# Set DOTk source and include directories
DOTk_SOURCE_DIR=/Users/miguelaguilo/locus/src/
DOTk_INCLUDE_DIR=/Users/miguelaguilo/locus/include/

# Set DOTk include directories
DOTk_EIGEN=$(DOTk_INCLUDE_DIR)DOTk_Eigen/
DOTk_VECTOR=$(DOTk_INCLUDE_DIR)DOTk_Vector/
DOTk_MATRIX=$(DOTk_INCLUDE_DIR)DOTk_Matrix/
DOTk_CCSA=$(DOTk_INCLUDE_DIR)DOTk_MethodCCSA/
DOTk_FUNCTOR=$(DOTk_INCLUDE_DIR)DOTk_Functor/
DOTk_HESSIAN=$(DOTk_INCLUDE_DIR)DOTk_Hessian/
DOTk_FACTORY=$(DOTk_INCLUDE_DIR)DOTk_Factory/
DOTk_VECTOR_SRC=$(DOTk_SOURCE_DIR)DOTk_Vector/
DOTk_MATRIX_SRC=$(DOTk_SOURCE_DIR)DOTk_Matrix/
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
	
# Set DOTk-MEX interface include directories
MEX_PTR=$(MEX_INCLUDE_DIR)DOTk_MexPointer/
MEX_PARSER=$(MEX_INCLUDE_DIR)DOTk_MexParser/
MEX_UTILITIES=$(MEX_INCLUDE_DIR)DOTk_MexUtilities/
MEX_FACTORIES=$(MEX_INCLUDE_DIR)DOTk_MexFactories/
MEX_FACTORIES_SRC=$(MEX_SOURCE_DIR)DOTk_MexFactories/
MEX_OPERATOR_API=$(MEX_INCLUDE_DIR)DOTk_MexOperatorsAPI/
MEX_OPERATOR_API_SRC=$(MEX_SOURCE_DIR)DOTk_MexOperatorsAPI/
MEX_ALG_TYPE_FO=$(MEX_INCLUDE_DIR)DOTk_MexAlgorithmTypeFO/
MEX_ALG_TYPE_OC=$(MEX_INCLUDE_DIR)DOTk_MexAlgorithmTypeOC/
MEX_DIAGNOSTICS=$(MEX_INCLUDE_DIR)DOTk_MexDiagnosticsTools/
MEX_ALG_TYPE_SQP=$(MEX_INCLUDE_DIR)DOTk_MexAlgorithmTypeSQP/
MEX_ALG_TYPE_CCSA=$(MEX_INCLUDE_DIR)DOTk_MexAlgorithmTypeCCSA/
MEX_ALG_TYPE_NEWTON=$(MEX_INCLUDE_DIR)DOTk_MexAlgorithmTypeNewton/
	
DOTK_OBJS=$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkNonLinearCG.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkMMA.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkGCMMA.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkIxSQPTypeTR.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkQuasiNewton.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkNewtonTypeLS.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkNewtonTypeTR.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkIxNewtonTypeLS.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkIxNewtonTypeTR.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkOptimalityCriteria.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkLinMoreTrustRegion.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkGradientProjection.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkIxLinMoreTrustRegion.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkKelleySachsTrustRegion.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmAPI/mexDOTkIxKelleySachsTrustRegion.o \
	$(MEX_SOURCE_DIR)DOTk_MexPointer/DOTk_MexArrayPtr.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeFO/DOTk_MexAlgorithmTypeFO.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeFO/DOTk_MexNonlinearCG.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeFO/DOTk_MexQuasiNewton.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeFO/DOTk_MexGradientProjection.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeOC/DOTk_MexOptimalityCriteria.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeSQP/DOTk_MexAlgorithmTypeSQP.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeSQP/DOTk_MexInexactTrustRegionSQP.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeCCSA/DOTk_MexMMA.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeCCSA/DOTk_MexGCMMA.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeCCSA/DOTk_MexMethodCCSA.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsTools/DOTk_MexDiagnostics.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsTools/DOTk_MexDiagnosticsTypeLP.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsTools/DOTk_MexDiagnosticsTypeNLP.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexAlgorithmTypeNewton.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexNewtonTypeLS.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexNewtonTypeTR.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexInexactNewtonTypeLS.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexInexactNewtonTypeTR.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexTrustRegionLinMore.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexSteihaugTointNewton.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexIxTrustRegionLinMore.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexTrustRegionKelleySachs.o \
	$(MEX_SOURCE_DIR)DOTk_MexAlgorithmTypeNewton/DOTk_MexIxTrustRegionKelleySachs.o \
	$(MEX_SOURCE_DIR)DOTk_MexOperatorsAPI/DOTk_MexInequalityConstraint.o \
	$(MEX_SOURCE_DIR)DOTk_MexOperatorsAPI/DOTk_MexEqualityConstraint.o \
	$(MEX_SOURCE_DIR)DOTk_MexOperatorsAPI/DOTk_MexObjectiveFunction.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexAlgorithmParser.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexMethodCcsaParser.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexQuasiNewtonParser.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexKrylovSolverParser.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexParseAlgorithmTypeSQP.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_SOURCE_DIR)DOTk_MexParser/DOTk_MexOptimalityCriteriaParser.o \
	$(MEX_SOURCE_DIR)DOTk_MexUtilities/DOTk_MexApiUtilities.o \
	$(MEX_SOURCE_DIR)DOTk_MexFactories/DOTk_MexHessianFactory.o \
	$(MEX_SOURCE_DIR)DOTk_MexFactories/DOTk_MexContainerFactory.o \
	$(MEX_SOURCE_DIR)DOTk_MexFactories/DOTk_MexNumDiffHessianFactory.o \
	$(MEX_SOURCE_DIR)DOTk_MexFactories/DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_SOURCE_DIR)DOTk_MexFactories/DOTk_MexFactoriesAlgorithmTypeFO.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsAPI/mexDOTkCheckFirstDerivativeTypeLP.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsAPI/mexDOTkCheckSecondDerivativeTypeLP.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsAPI/mexDOTkCheckFirstDerivativeTypeNLP.o \
	$(MEX_SOURCE_DIR)DOTk_MexDiagnosticsAPI/mexDOTkCheckSecondDerivativeTypeNLP.o \
	$(DOTk_SOURCE_DIR)DOTk_LineSearchAlgorithmsDataMng/DOTk_OptimizationDataMng.o \
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
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_MultiVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_PrimalVector.o \
	$(DOTk_SOURCE_DIR)DOTk_Vector/DOTk_ParallelUtils.o \
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
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_ProjectedSteihaugTointPcg.o \
	$(DOTk_SOURCE_DIR)DOTk_SteihaugToint/DOTk_SteihaugTointProjGradStep.o \
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

MEX_OC=$(MEX_OBJECT_DIR)mexDOTkOptimalityCriteria.o \
	$(MEX_OBJECT_DIR)DOTk_MexOptimalityCriteriaParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexInequalityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexOptimalityCriteria.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_OptimalityCriteriaRoutineMng.o \
	$(MEX_OBJECT_DIR)DOTk_OptimalityCriteriaDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_OptimalityCriteria.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_MMA=$(MEX_OBJECT_DIR)mexDOTkMMA.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexMethodCcsaParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexMethodCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_MexMMA.o \
	$(MEX_OBJECT_DIR)DOTk_DualObjectiveFunctionMMA.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_ScaleParametersNLCG.o \
	$(MEX_OBJECT_DIR)DOTk_DataMngNonlinearCG.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraints.o \
	$(MEX_OBJECT_DIR)DOTk_DualSolverCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_DualSolverNLCG.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SubProblemCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_SubProblemMMA.o \
	$(MEX_OBJECT_DIR)DOTk_AlgorithmCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTK_MethodCcsaIO.o \
	$(MEX_OBJECT_DIR)DOTk_DataMngCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_UtilsCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_RowMatrix.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_GCMMA=$(MEX_OBJECT_DIR)mexDOTkGCMMA.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexMethodCcsaParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexMethodCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_MexGCMMA.o \
	$(MEX_OBJECT_DIR)DOTk_DualObjectiveFunctionMMA.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_ScaleParametersNLCG.o \
	$(MEX_OBJECT_DIR)DOTk_DataMngNonlinearCG.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraints.o \
	$(MEX_OBJECT_DIR)DOTk_SubProblemGCMMA.o \
	$(MEX_OBJECT_DIR)DOTk_DualSolverCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_DualSolverNLCG.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SubProblemCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_AlgorithmCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTK_MethodCcsaIO.o \
	$(MEX_OBJECT_DIR)DOTk_DataMngCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_UtilsCCSA.o \
	$(MEX_OBJECT_DIR)DOTk_RowMatrix.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_NONLINEAR_CG=$(MEX_OBJECT_DIR)mexDOTkNonLinearCG.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexNumDiffHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeFO.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexNonlinearCG.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderBackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_NumericallyDifferentiatedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiatonFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderLineSearchAlgIO.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiation.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_CentralFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderAlgorithm.o \
	$(MEX_OBJECT_DIR)DOTk_NonlinearCGFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchFactory.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ConjugateDescent.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirection.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_HestenesStiefel.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FletcherReeves.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_DaiYuanHybrid.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_PolakRibiere.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_NonlinearCG.o \
	$(MEX_OBJECT_DIR)DOTk_PerryShanno.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhang.o \
	$(MEX_OBJECT_DIR)DOTk_LiuStorey.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_DaiYuan.o \
	$(MEX_OBJECT_DIR)DOTk_DaiLiao.o \
	$(MEX_OBJECT_DIR)DOTk_Daniels.o \
	$(MEX_OBJECT_DIR)DOTk_Functor.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_QUASI_NEWTON=$(MEX_OBJECT_DIR)mexDOTkQuasiNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeFO.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderLineSearchAlgIO.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InverseHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchQuasiNewton.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderAlgorithm.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchFactory.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_SR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_TR_LIN_MORE=$(MEX_OBJECT_DIR)mexDOTkLinMoreTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexSteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexTrustRegionLinMore.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointProjGradStep.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiation.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_CentralFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointLinMore.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointSolver.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraints.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointPcg.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_Preconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_IxTR_LIN_MORE=$(MEX_OBJECT_DIR)mexDOTkIxLinMoreTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexNumDiffHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexIxTrustRegionLinMore.o \
	$(MEX_OBJECT_DIR)DOTk_MexSteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderBackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_NumericallyDifferentiatedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiatonFactory.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointProjGradStep.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiation.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_CentralFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointLinMore.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointSolver.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraints.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointPcg.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_Preconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Functor.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o
	
MEX_TR_KELLEY_SACHS=$(MEX_OBJECT_DIR)mexDOTkKelleySachsTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexTrustRegionKelleySachs.o \
	$(MEX_OBJECT_DIR)DOTk_MexSteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedSteihaugTointPcg.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointKelleySachs.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KelleySachsStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraints.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointPcg.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_Preconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o
	
MEX_IxTR_KELLEY_SACHS=$(MEX_OBJECT_DIR)mexDOTkIxKelleySachsTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexIxTrustRegionKelleySachs.o \
	$(MEX_OBJECT_DIR)DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexNumDiffHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexSteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderBackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_NumericallyDifferentiatedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiatonFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedSteihaugTointPcg.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiation.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointKelleySachs.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_CentralFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointNewton.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KelleySachsStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionStepMng.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraints.o \
	$(MEX_OBJECT_DIR)DOTk_SteihaugTointPcg.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_Preconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Functor.o \
	$(MEX_OBJECT_DIR)DOTk_Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_NEWTON_TYPE_LS=$(MEX_OBJECT_DIR)mexDOTkNewtonTypeLS.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexNewtonTypeLS.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_SecantLeftPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverStoppingCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TangentialSubProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjectionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_UpperTriangularDirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemPrecFactory.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchInexactNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGenConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTK_SecantLeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemLeftPrec.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchInexactNewton.o \
	$(MEX_OBJECT_DIR)DOTk_InexactNewtonAlgorithms.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjGradDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InverseHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjLeftPrecCgDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGenMinResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNEqDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjection.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ArnoldiProjection.o \
	$(MEX_OBJECT_DIR)DOTk_RelativeCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystem.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FixedCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_DirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolver.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNR.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNE.o \
	$(MEX_OBJECT_DIR)DOTk_GramSchmidt.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGCR.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCR.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGMRES.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_NEWTON_TYPE_TR=$(MEX_OBJECT_DIR)mexDOTkNewtonTypeTR.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexNewtonTypeTR.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_SecantLeftPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverStoppingCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TangentialSubProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjectionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UpperTriangularDirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemPrecFactory.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionInexactNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGenConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTK_SecantLeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionInexactNewton.o \
	$(MEX_OBJECT_DIR)DOTk_DoubleDoglegTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemLeftPrec.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InexactNewtonAlgorithms.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjGradDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InverseHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjLeftPrecCgDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionMngTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionMngTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGenMinResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNEqDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjection.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ArnoldiProjection.o \
	$(MEX_OBJECT_DIR)DOTk_RelativeCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_DoglegTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystem.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FixedCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_DirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolver.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNR.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNE.o \
	$(MEX_OBJECT_DIR)DOTk_GramSchmidt.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGCR.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCR.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGMRES.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_IxSQP_TYPE_TR=$(MEX_OBJECT_DIR)mexDOTkIxSQPTypeTR.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexParseAlgorithmTypeSQP.o \
	$(MEX_OBJECT_DIR)DOTk_MexInexactTrustRegionSQP.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeSQP.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_SecantLeftPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_SequentialQuadraticProgramming.o \
	$(MEX_OBJECT_DIR)DOTk_InexactTrustRegionSqpSolverMng.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverStoppingCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TangentialSubProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjectionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UpperTriangularDirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemPrecFactory.o \
	$(MEX_OBJECT_DIR)DOTk_TangentialProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGenConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTK_SecantLeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_QuasiNormalProbCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_InexactTrustRegionSqpIO.o \
	$(MEX_OBJECT_DIR)DOTk_SqpDualProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_DoubleDoglegTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemLeftPrec.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjGradDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionMngTypeELP.o \
	$(MEX_OBJECT_DIR)DOTk_InexactTrustRegionSQP.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InverseHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjLeftPrecCgDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGenMinResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNEqDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjection.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_ArnoldiProjection.o \
	$(MEX_OBJECT_DIR)DOTk_RelativeCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_DoglegTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystem.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeELP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeENP.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_BFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FixedCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_DirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolver.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNR.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNE.o \
	$(MEX_OBJECT_DIR)DOTk_GramSchmidt.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGCR.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCR.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGMRES.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_IxNEWTON_TYPE_LS=$(MEX_OBJECT_DIR)mexDOTkIxNewtonTypeLS.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexNumDiffHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexInexactNewtonTypeLS.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderBackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_NumericallyDifferentiatedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SecantLeftPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiatonFactory.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverStoppingCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TangentialSubProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjectionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_UpperTriangularDirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemPrecFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchInexactNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGenConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTK_SecantLeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiation.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_CentralFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemLeftPrec.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchInexactNewton.o \
	$(MEX_OBJECT_DIR)DOTk_InexactNewtonAlgorithms.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjGradDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InverseHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjLeftPrecCgDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchMngTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGenMinResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNEqDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjection.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ArnoldiProjection.o \
	$(MEX_OBJECT_DIR)DOTk_RelativeCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystem.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchStep.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FixedCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_DirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolver.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNR.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNE.o \
	$(MEX_OBJECT_DIR)DOTk_GramSchmidt.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGCR.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCR.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGMRES.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Functor.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_IxNEWTON_TYPE_TR=$(MEX_OBJECT_DIR)mexDOTkIxNewtonTypeTR.o \
	$(MEX_OBJECT_DIR)DOTk_MexFactoriesAlgorithmTypeGB.o \
	$(MEX_OBJECT_DIR)DOTk_MexFiniteDiffNumIntgParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexNumDiffHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmTypeNewton.o \
	$(MEX_OBJECT_DIR)DOTk_MexInexactNewtonTypeTR.o \
	$(MEX_OBJECT_DIR)DOTk_MexKrylovSolverParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexQuasiNewtonParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderBackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ThirdOrderForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_NumericallyDifferentiatedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SecantLeftPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiatonFactory.o \
	$(MEX_OBJECT_DIR)DOTk_BacktrackingCubicInterpolation.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverStoppingCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TangentialSubProblemCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionAlgorithmsDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjectionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_UpperTriangularDirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditionerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectionAlongFeasibleDir.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemPrecFactory.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionInexactNewtonIO.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessianTypeCNP.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperatorFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGenConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelBackwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTK_SecantLeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionInexactNewton.o \
	$(MEX_OBJECT_DIR)DOTk_NumericalDifferentiation.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_CentralFiniteDifference.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystemLeftPrec.o \
	$(MEX_OBJECT_DIR)DOTk_GoldenSectionLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelCentralDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelForwardDiffGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InexactNewtonAlgorithms.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjGradDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DoubleDoglegTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecConjResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_BarzilaiBorweinHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BackwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraintFactory.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionMngTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionMngTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_CentralDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_ForwardDifferenceGrad.o \
	$(MEX_OBJECT_DIR)DOTk_InverseHessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ProjLeftPrecCgDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGenMinResDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNEqDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_HagerZhangLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_OrthogonalProjection.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_GoldsteinLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_SecondOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolverFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ProjectedLeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_RightPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPreconditioner.o \
	$(MEX_OBJECT_DIR)DOTk_FirstOrderOperator.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedHessian.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegionFactory.o \
	$(MEX_OBJECT_DIR)DOTk_FeasibleDirection.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearchFactory.o \
	$(MEX_OBJECT_DIR)DOTk_ArnoldiProjection.o \
	$(MEX_OBJECT_DIR)DOTk_RelativeCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_DoglegTrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_GradBasedIoUtils.o \
	$(MEX_OBJECT_DIR)DOTk_ArmijoLineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_AssemblyManager.o \
	$(MEX_OBJECT_DIR)DOTk_AugmentedSystem.o \
	$(MEX_OBJECT_DIR)DOTk_UserDefinedGrad.o \
	$(MEX_OBJECT_DIR)DOTk_BoundConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_GradientTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeULP.o \
	$(MEX_OBJECT_DIR)DOTk_RoutinesTypeUNP.o \
	$(MEX_OBJECT_DIR)DOTk_LinearOperator.o \
	$(MEX_OBJECT_DIR)DOTk_VariablesUtils.o \
	$(MEX_OBJECT_DIR)DOTk_HessianFactory.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_BFGSInvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_FixedCriterion.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_SR1InvHessian.o \
	$(MEX_OBJECT_DIR)DOTk_DirectSolver.o \
	$(MEX_OBJECT_DIR)DOTk_KrylovSolver.o \
	$(MEX_OBJECT_DIR)DOTk_LBFGSHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNR.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCGNE.o \
	$(MEX_OBJECT_DIR)DOTk_TrustRegion.o \
	$(MEX_OBJECT_DIR)DOTk_GramSchmidt.o \
	$(MEX_OBJECT_DIR)DOTk_LDFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_LSR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecGCR.o \
	$(MEX_OBJECT_DIR)DOTk_DFPHessian.o \
	$(MEX_OBJECT_DIR)DOTk_SR1Hessian.o \
	$(MEX_OBJECT_DIR)DOTk_LineSearch.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCG.o \
	$(MEX_OBJECT_DIR)DOTk_LeftPrecCR.o \
	$(MEX_OBJECT_DIR)DOTk_MathUtils.o \
	$(MEX_OBJECT_DIR)DOTk_PrecGMRES.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Functor.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_FRST_DIGNSTCS_TYPELP=$(MEX_OBJECT_DIR)mexDOTkCheckFirstDerivativeTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexInequalityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnosticsTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnostics.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_DerivativeDiagnosticsTool.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_SECND_DIGNSTCS_TYPELP=$(MEX_OBJECT_DIR)mexDOTkCheckSecondDerivativeTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexInequalityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnosticsTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnostics.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_DerivativeDiagnosticsTool.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsTypeLP.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_FRST_DIGNSTCS_TYPENLP=$(MEX_OBJECT_DIR)mexDOTkCheckFirstDerivativeTypeNLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexInequalityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnosticsTypeNLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnostics.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_DerivativeDiagnosticsTool.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsInequalityTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsObjectiveTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsEqualityTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_NonLinearProgrammingUtils.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

MEX_SECND_DIGNSTCS_TYPENLP=$(MEX_OBJECT_DIR)mexDOTkCheckSecondDerivativeTypeNLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexInequalityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexEqualityConstraint.o \
	$(MEX_OBJECT_DIR)DOTk_MexObjectiveFunction.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnosticsTypeNLP.o \
	$(MEX_OBJECT_DIR)DOTk_MexContainerFactory.o \
	$(MEX_OBJECT_DIR)DOTk_MexAlgorithmParser.o \
	$(MEX_OBJECT_DIR)DOTk_MexApiUtilities.o \
	$(MEX_OBJECT_DIR)DOTk_MexDiagnostics.o \
	$(MEX_OBJECT_DIR)DOTk_MexArrayPtr.o \
	$(MEX_OBJECT_DIR)DOTk_DerivativeDiagnosticsTool.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsInequalityTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsObjectiveTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_DiagnosticsEqualityTypeNP.o \
	$(MEX_OBJECT_DIR)DOTk_NonLinearProgrammingUtils.o \
	$(MEX_OBJECT_DIR)DOTk_DescentDirectionTools.o \
	$(MEX_OBJECT_DIR)DOTk_OptimizationDataMng.o \
	$(MEX_OBJECT_DIR)DOTk_ParallelUtils.o \
	$(MEX_OBJECT_DIR)DOTk_Variable.o \
	$(MEX_OBJECT_DIR)DOTk_Primal.o \
	$(MEX_OBJECT_DIR)DOTk_Control.o \
	$(MEX_OBJECT_DIR)DOTk_State.o \
	$(MEX_OBJECT_DIR)DOTk_Dual.o

.cpp.o:
 
	$(MEX) -largeArrayDims -outdir $(MEX_OBJECT_DIR) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) $(ALL_INCLUDE_DIR) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) -c $@ $<

locus: $(DOTK_OBJS)

MEX:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_OC)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_MMA)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_GCMMA)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_TR_LIN_MORE)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_IxTR_LIN_MORE)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_TR_KELLEY_SACHS)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_IxTR_KELLEY_SACHS)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_NONLINEAR_CG)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_QUASI_NEWTON)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_IxSQP_TYPE_TR)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_NEWTON_TYPE_LS)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_NEWTON_TYPE_TR)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_IxNEWTON_TYPE_LS)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_IxNEWTON_TYPE_TR)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_FRST_DIGNSTCS_TYPELP)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_SECND_DIGNSTCS_TYPELP)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_FRST_DIGNSTCS_TYPENLP)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) -outdir $(EXEDIR) $(MEX_SECND_DIGNSTCS_TYPENLP)

all:
	make clean
	mkdir obj 
	mkdir exe
	make -f Makefile locus
	make -f Makefile MEX
	
clean:
	$(RM) -rf $(MEX_OBJECT_DIR) $(EXEDIR)
	find . -name "*.o" -exec rm {} \;
	
