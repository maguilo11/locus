
function [output] = main

clc;

global GLB_INVP;

%%%%%%%%%%%%%%%%%%%%%%%%%% Paths to dependencies %%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ./source/new/;
addpath ./source/new/interface/;
addpath /Users/miguelaguilo/locus/intrelab;
addpath /Users/miguelaguilo/locus/femlab/mesh_tools/;
addpath /Users/miguelaguilo/Research/femlab/utilities/;
addpath /Users/miguelaguilo/locus/femlab/algorithm/gcmma/;
%%%%%%%%%%%%%%%%%%%%%%%%%% Paths to dependencies %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Problem setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
problem_t = 'compliance';
objective = objectiveFunction;
equality = equalityConstraint;
inequality = inequalityConstraint;
switch problem_t
    case 'basic'
        inequality.number_inequalities = 2;
        % Set Control Space Dimension and Initial Guess
        number_controls = 9;
        initial_control = 0.25 * ones(number_controls,1);
    case 'cantilever'
        inequality.number_inequalities = 1;
        % Set Control Space Dimension and Initial Guess
        number_controls = 5;
        initial_control = 5 * ones(number_controls,1);
        control_lower_bound = 1e-3.*ones(number_controls,1);
        control_upper_bound = 1e1.*ones(number_controls,1);
    case 'compliance'
        mesh_file = '/Users/miguelaguilo/locus/femlab/mesh_tools/data/lbracket_2D_quad.exo';
        [GLB_INVP] = driverTOPT(mesh_file);
        inequality.number_inequalities = 1;
        number_controls = GLB_INVP.nVertGrid;
        one = ones(GLB_INVP.nVertGrid,1);
        GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*one);
        initial_control = GLB_INVP.VolumeFraction * ones(number_controls,1);
        GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
        GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
        [state] = equality.solve(one);
        StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
        K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
        GLB_INVP.theta = 0.5 / (state'*K*state);
        control_upper_bound = ones(number_controls,1);
        control_lower_bound = zeros(number_controls,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Problem setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Solve nonlinear programming problem with GCMMA
tol = 5e-6;
max_outer_itr = 100;
a_coefficients = zeros(1+inequality.number_inequalities,1);
a_coefficients(1) = 1;
d_coefficients = ones(inequality.number_inequalities,1);
c_coefficients = 1e3*ones(inequality.number_inequalities,1);
[output] = ...
    gcmma(objective, equality, inequality, initial_control, ...
    control_lower_bound, control_upper_bound,...
    a_coefficients, d_coefficients, c_coefficients,max_outer_itr, tol);
 show(GLB_INVP.mesh.t, GLB_INVP.mesh.p,output.control);

end
