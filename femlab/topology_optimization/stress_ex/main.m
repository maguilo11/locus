
function [output] = main

clc;

global GLB_INVP;

%%%%%%%%%%%%%%%%%%%%%%%%%% Paths to dependencies %%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ./src/nova/;
addpath ./src/nova/interface/;
addpath /Users/miguelaguilo/locus/intrelab/;
addpath /Users/miguelaguilo/locus/femlab/mesh_tools/;
addpath /Users/miguelaguilo/Research/femlab/utilities/;
addpath /Users/miguelaguilo/locus/femlab/algorithm/gcmma/;
%%%%%%%%%%%%%%%%%%%%%%%%%% Paths to dependencies %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Operators %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
derivative_check = false;
objective = objectiveFunction;
equality = equalityConstraint;
inequality = inequalityConstraint;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Operators %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% Read exodus file (mesh) %%%%%%%%%%%%%%%%%%%%%%%%%
mesh_file = '/Users/miguelaguilo/locus/femlab/mesh_tools/data/lbracket_2D_quad.exo';
%%%%%%%%%%%%%%%%%%%%%%%%% Read exodus file (mesh) %%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Physics problem setup %%%%%%%%%%%%%%%%%%%%%%%%%%
[GLB_INVP] = driverTOPT(mesh_file);
one = ones(GLB_INVP.nVertGrid,1);
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*one);
initial_control = GLB_INVP.VolumeFraction * ones(GLB_INVP.nVertGrid,1);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
%avg_cell_weight = sum(GLB_INVP.ElemVolume) / size(GLB_INVP.ElemVolume,1);
%[GLB_INVP.Filter] = Filter(GLB_INVP.mesh, avg_cell_weight, GLB_INVP.filter_radius);
GLB_INVP.Filter = speye(GLB_INVP.nVertGrid);
%%%%%%%%%%%%%%%%%%%%%%%%%% Physics problem setup %%%%%%%%%%%%%%%%%%%%%%%%%%

if(derivative_check == true)
    %%%%%%%%%%%%%%%%% Finite difference derivative check %%%%%%%%%%%%%%%%%%
    [error] = checkGradient(objective,equality,initial_control);
    plot(error);
    %%%%%%%%%%%%%%%%% Finite difference derivative check %%%%%%%%%%%%%%%%%%
else
    %%%%%%%%%%%%%%%%%%%%% Solve optimization problem %%%%%%%%%%%%%%%%%%%%%%
    tol = 1e-8;
    max_outer_itr = 200;
    inequality.number_inequalities = 1;
    number_controls = GLB_INVP.nVertGrid;
    a_coefficients = zeros(1+inequality.number_inequalities,1);
    a_coefficients(1) = 1;
    d_coefficients = ones(inequality.number_inequalities,1);
    c_coefficients = 1e4*ones(inequality.number_inequalities,1);
    control_upper_bound = ones(number_controls,1);
    control_lower_bound = zeros(number_controls,1);
    
    [output] = gcmma(objective, equality, inequality, initial_control, ...
        control_lower_bound, control_upper_bound,a_coefficients, ...
        d_coefficients, c_coefficients,max_outer_itr, tol);
    
    %%%%%%%%%%%%%%%%%%%%% Solve optimization problem %%%%%%%%%%%%%%%%%%%%%%
    show(GLB_INVP.mesh.t, GLB_INVP.mesh.p,output.control);
end

end
%
%%%%%%%%%%%%%%%%%%%%%%%%% FINITE DIFFERENCE CHECK %%%%%%%%%%%%%%%%%%%%%%%%%
%
function [error] = checkGradient(objective,equality,control)

rng(1);
step = randn(size(control));
% Compute gradient
[state] = equality.solve(control);
[grad] = objective.gradient(state,control);
trueGradDotStep = grad'*step;
error = zeros(10,1);
for i=1:10
    epsilon = 1/(10^(i-1));
    trial_control = control + (epsilon.*step);
    [state] = equality.solve(trial_control);
    [FvalP] = objective.evaluate(state,trial_control);
    trial_control = control - (epsilon.*step);
    [state] = equality.solve(trial_control);
    [FvalM] = objective.evaluate(state,trial_control);
    finiteDiffAppx = (FvalP - FvalM) / (2*epsilon);
    error(i) = abs(trueGradDotStep-finiteDiffAppx);
end

end