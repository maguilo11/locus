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
%% TEST ORTHOGONAL DECOMPOSITION MEX INTERFACE %%
%% TEST SINGULAR VALUE DECOMPOSITION MEX INTERFACE %%
end