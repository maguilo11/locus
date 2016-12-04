function [Output] = ccsa()

clc;

fprintf('\n*** Conservative Convex Separable Approximations (CCSA) Method ***\n');

addpath ./topo/;
addpath ./topo_ccsa/;
addpath /Users/miguelaguilo/dotk/matlab/exe/;
addpath /Users/miguelaguilo/Research/intrelab;
addpath /Users/miguelaguilo/dotk/matlab/mfiles/;
addpath /Users/miguelaguilo/Research/femlab/tools/;
%addpath /Users/miguelaguilo/dotk/matlab/mfiles/MMA/;

problem_t = 'topo';
global GLB_INVP;

switch problem_t
    case 'basic'
        Inputs.ProblemType = 'CLP';
        Inputs.NumberDuals = 1;
        Inputs.NumberControls = 5;
        Inputs.InitialControl = 5 * ones(1,Inputs.NumberControls);
        Inputs.ControlLowerBounds = 1e-3 * ones(1,Inputs.NumberControls);
        Inputs.ControlUpperBounds = 10 * ones(1,Inputs.NumberControls);
        
        [Options, Operators] = setAlgorithmGCMMA(Inputs);
        [Output] = mexDOTkGCMMA(Options, Operators);
    case 'topo'
        Inputs.ProblemType = 'CNLP';
        [GLB_INVP] = driverTOPT;
        Inputs.NumberDuals = 1;
        Inputs.NumberStates = GLB_INVP.spaceDim * GLB_INVP.nVertGrid;
        Inputs.NumberControls = GLB_INVP.nVertGrid;
        Inputs.ControlUpperBounds = ones(Inputs.NumberControls,1);
        Inputs.ControlLowerBounds = 1e-3.*ones(Inputs.NumberControls,1);
        one = ones(GLB_INVP.nVertGrid,1);
        GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*one);
        Inputs.InitialControl = GLB_INVP.VolumeFraction * ones(Inputs.NumberControls,1);
        
        [Options, Operators] = setAlgorithmGCMMA(Inputs);
        
        GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
        GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
        [state] = Operators.EqualityConstraint.solve(one);
        StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
        K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
        GLB_INVP.theta = 0.5 / (state'*K*state);
        
        [Output] = mexDOTkGCMMA(Options, Operators);
end

end