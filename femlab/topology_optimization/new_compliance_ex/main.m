
function [output] = main

clc;

global GLB_INVP;

%%%%%%%%%%%%%%%%%%%%%%%%%% Paths to dependencies %%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ./source/new/;
addpath ./source/new/interface/;
addpath /Users/miguelaguilo/locus/intrelab/;
addpath /Users/miguelaguilo/locus/femlab/utilities/;
addpath /Users/miguelaguilo/locus/femlab/mesh_tools/;
addpath /Users/miguelaguilo/locus/femlab/algorithm/gcmma/;
%%%%%%%%%%%%%%%%%%%%%%%%%% Paths to dependencies %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Problem setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
multi_material = true;
derivative_check = false;
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
        % Set problem parameters and data dependencies 
        mesh_file = '/Users/miguelaguilo/locus/femlab/mesh_tools/data/lbracket_2D_quad.exo';
        [GLB_INVP] = driverTOPT(mesh_file,multi_material);
        GLB_INVP.PenaltyModel = modifiedSIMP;
        GLB_INVP.InterpolationRule = summationRule;
        inequality.number_inequalities = length(GLB_INVP.VolumeFraction);
        
        % Set initial guess
        data = ones(GLB_INVP.nVertGrid,1);
        GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*data);
        number_controls = GLB_INVP.num_materials * GLB_INVP.nVertGrid;
        initial_control = GLB_INVP.VolumeFraction .* ones(number_controls,1);
        
        % Precompute stiffness matrices
        GLB_INVP.CellStifsnessMat =...
            computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
        
        % Build filter operator
        avg_cell_weight = ...
            sum(GLB_INVP.ElemVolume) / size(GLB_INVP.ElemVolume,1);
        [GLB_INVP.Filter] = ...
            Filter(GLB_INVP.mesh, avg_cell_weight, GLB_INVP.filter_radius);
        
        % Compute objective function scale factor
        data = sum(GLB_INVP.VolumeFraction) .* ones(number_controls,1);
        [state] = equality.solve(data);
        CellStiffnessMatrix = ...
            zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells);
        for material_index=1:GLB_INVP.num_materials
            CellStiffnessMatrix = CellStiffnessMatrix + ...
                GLB_INVP.CellStifsnessMat(:,:,:,material_index);
        end
        StiffnessMatrix = ...
            reshape(CellStiffnessMatrix, 1, numel(CellStiffnessMatrix));
        K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffnessMatrix);
        GLB_INVP.theta = 0.5 / (state'*K*state);
        control_upper_bound = ones(number_controls,1);
        control_lower_bound = zeros(number_controls,1);
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Problem setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if(derivative_check == true)
    %%%%%%%%%%%%%%%%% Finite difference derivative check %%%%%%%%%%%%%%%%%%
    [error] = checkGradient(objective,equality,initial_control);
    semilogy(error);
else
    %%%%%%%%%%% Solve nonlinear programming problem with GCMMA %%%%%%%%%%%%
    tol = 1e-12;
    max_outer_itr = 100;
    a_coefficients = zeros(1+inequality.number_inequalities,1);
    a_coefficients(1) = 1;
    d_coefficients = ones(inequality.number_inequalities,1);
    c_coefficients = 1e3*ones(inequality.number_inequalities,1);
    tic
    [output] = ...
        gcmma(objective, equality, inequality, initial_control, ...
        control_lower_bound, control_upper_bound,...
        a_coefficients, d_coefficients, c_coefficients,max_outer_itr, tol);
    toc
    if(GLB_INVP.multi_material == true)
        show(GLB_INVP.mesh.t, GLB_INVP.mesh.p,output.control(1:GLB_INVP.nVertGrid));
        show(GLB_INVP.mesh.t, GLB_INVP.mesh.p,output.control(1+GLB_INVP.nVertGrid:end));
    else
        show(GLB_INVP.mesh.t, GLB_INVP.mesh.p,output.control(1:GLB_INVP.nVertGrid));
    end
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