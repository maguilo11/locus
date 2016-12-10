function utest
clc;
addpath('exe/');
addpath('mfiles/');
%% TEST VECTOR MEX INTERFACE %%
TRROM_MxVectorTest;
%% TEST MATRIX MEX INTERFACE %%
TRROM_MxMatrixTest;
%% TEST OBJECTIVE MEX INTERFACE %%
obj = objective;
TRROM_MxReducedObjectiveOperatorsTest(obj);
%% TEST INEQUALITY CONSTRAINT MEX INTERFACE %%
inq = inequality;
TRROM_MxInequalityOperatorsTest(inq);
%% TEST REDUCED BASIS PDE CONSTRAINT MEX INTERFACE %%
pde = equality;
TRROM_MxReducedBasisPDETest(pde);
%% TEST DIRECT SOLVER MEX INTERFACE %%
%A = [1 5 0 0 0; 0 2 8 0 0; 0 0 3 9 0; 0 0 0 4 10; 0 0 0 0 5];
%b = [1; 2; 3; 4; 5];
TRROM_MxDirectSolverTest;
%% TEST ORTHOGONAL DECOMPOSITION MEX INTERFACE %%
A = [-1 -1 1; 1 3 3; -1 -1 5; 1 3 7];
[~,~,~]=TRROM_MxOrthogonalDecompositionTest(A);
%% TEST SINGULAR VALUE DECOMPOSITION MEX INTERFACE %%
A = [1 0 1; -1 -2 0; 0 1 -1];
[~,~,~]=TRROM_MxSpectralDecompositionTest(A);
%% TEST BRAND MATRIX FACTORY MEX INTERFACE %%
TRROM_MxBrandMatrixFactoryTest;
%% TEST BRAND ALGORITHM MEX INTERFACE %%
TRROM_MxLowRankSVDTest;
%% TEST DISCRETE EMPIRICAL INTERPOLATION METHOD MEX INTERFACE
TRROM_MxDiscreteEmpiricalInterpolationTest;
%% TEST LINEAR ALGEBRA FACTORY MEX INTERFACE
TRROM_MxLinearAlgebraFactoryTest;
end