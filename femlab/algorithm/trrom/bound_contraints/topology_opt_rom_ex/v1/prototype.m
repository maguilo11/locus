function [Primal,OptDataMng,Results] = ...
    prototype(stepTol,actualRedTol,maxOptItr,maxSubProbItr)

% Prototype for surrogate based topology optimization
% NOTES:
% - Based on C.T. Kelley and E.W. Sachs 1999 paper: a trust region method
% for parabolic boundary control problems
clc;

% Problem interface directories
addpath ./interface/;
addpath /Users/miguelaguilo/Research/intrelab;
addpath /Users/miguelaguilo/Research/femlab/tools/;
addpath /Users/miguelaguilo/Research/intrelab/mesh2/;

global GLB_INVP;

fprintf('\n*** Structural Topology Optimization ***\n');

% Domain specifications
Domain.xmin = 0;   % min dim in x-dir
Domain.xmax = 1.5; % max dim in x-dir
Domain.ymin = 0;   % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 45;    % num intervals in x-dir
Domain.ny = 30;    % num intervals in y-dir
% Enable derivative check
derivativeCheck = false;
% regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV';
gamma = 1e-1;
beta = 0.5*(Domain.xmax / (Domain.nx));
% Set random number generator seed
rng(0,'v5normal');
% Set forcing function
forceFunc=@(struc)generateNodalForce(struc);
% Generate finite element specific quantities
[GLB_INVP] = generateParams(Domain,forceFunc);
% Set control space dimension
Primal = [];
Primal.Control.numControls = GLB_INVP.nVertGrid;
Primal.Control.current = ones(GLB_INVP.nVertGrid,1);
Primal.Control.upperBound = ones(size(Primal.Control.current));
Primal.Control.lowerBound = zeros(size(Primal.Control.current));
% Set low fidelity model parameters
Primal.State.LFM = 0;
Primal.State.HFM = 0;
Primal.State.basis = 0;
Primal.State.numSnapshots = 0;
Primal.State.numModelUpdates = 0;
Primal.State.updateModel = false;
Primal.State.highFidelity = true;
numLoadCases = size(GLB_INVP.force,2);
numRows = GLB_INVP.spaceDim*GLB_INVP.nVertGrid;
numColumns = numLoadCases*maxOptItr*maxSubProbItr;
Primal.State.storage = zeros(numRows,numColumns);
OptDataMng.currentObjFuncInexactness = -1e10;
% Get objective and PDE operators
Operators = [];
Operators.equality = equalityConstraint;
Operators.objective = objectiveFunction;
% Set inverse problem specific parameters 
GLB_INVP.reg = reg;
GLB_INVP.beta = beta;
GLB_INVP.gamma = gamma;
GLB_INVP.model_t = 'simp';
GLB_INVP.VolumeFraction = 0.3;
GLB_INVP.HessianType = 'GaussNewton';
% Helmholtz filter radius of influence
GLB_INVP.epsilon = 1/1e3;
% Set topology optimization specific parameters
GLB_INVP.SimpPenalty = 3;
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*Primal.Control.current);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
% Set normalization constants
Primal.State.numStates = GLB_INVP.nVertGrid * GLB_INVP.spaceDim;
GLB_INVP.alpha = 1e0 / (sum(GLB_INVP.Ms*(GLB_INVP.VolumeFraction*Primal.Control.current))^2);
[Primal.State] = Operators.equality.solve(Primal.State,Primal.Control);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = zeros(5,1);
for index=1:size(GLB_INVP.force,2)
    GLB_INVP.theta(index) = 0.5 / ...
        (Primal.State.current(:,index)'*(K*Primal.State.current(:,index)));
end
clear K;
% Set trust region algorithm parameters
OptDataMng.etaF = 1;
OptDataMng.alpha = 1e-2;
OptDataMng.stepTol = stepTol;
OptDataMng.maxOptItr = maxOptItr;
OptDataMng.actualRedTol = actualRedTol;
OptDataMng.maxSubProbItr = maxSubProbItr;
% Trust region parameters
OptDataMng.maxTrustRegion = 1e4;
OptDataMng.trustRegionExpansion = 2;
OptDataMng.trustRegionMidBound = 0.25;
OptDataMng.trustRegionReduction = 0.25;
OptDataMng.trustRegionLowerBound = 0.1;
OptDataMng.trustRegionUpperBound = 0.75;
% Construct active and inactive sets
OptDataMng.active = zeros(size(Primal.Control.current));
OptDataMng.inactive = ones(size(Primal.Control.current));
% Set initial guess for control
Primal.Control.current = ...
    GLB_INVP.VolumeFraction * ones(Primal.Control.numControls,1);
% Solve problem
proctime = cputime;
walltime = tic;
if(derivativeCheck == true)
    [Results.gradError,Primal,OptDataMng.currentDual] = ...
        checkGradient(Operators,Primal);
    [Results.hessError] = checkHessian(Operators,OptDataMng,Primal);
else
    [Primal,OptDataMng,Results] = getMin(Operators,Primal,OptDataMng);
end
proctime = cputime - proctime;
walltime = toc(walltime);

fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', walltime);

% Plot Results
plotResults(-1*Primal.Control.current);
save('primal.mat','Primal');
save('results.mat','Results');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,OptDataMng,OptData] = getMin(Operators,Primal,OptDataMng)

% Compute objective function
Primal.Control.old = Primal.Control.upperBound;
[OptDataMng.currentFval,Primal] = objective(Operators,Primal);
OptDataMng.numGradientCheck = 0;
OptDataMng.actualReduction = OptDataMng.currentFval;
OptDataMng.currentHighFidelFval = OptDataMng.currentFval;
% Construct State Sotrage
OptDataMng.inactive = ones(size(Primal.Control.current));
% Compute gradient
[OptDataMng.currentGrad,OptDataMng.currentDual] = ...
    gradient(Operators,Primal);
% Solve optimization problem
OptData.Fval = zeros(OptDataMng.maxOptItr,1);
OptData.optimality = zeros(OptDataMng.maxOptItr,1);
OptData.actualReduction = zeros(OptDataMng.maxOptItr,1);
OptData.nonStatinarityMeas = zeros(OptDataMng.maxOptItr,1);
OptDataMng.trustRegionRadius = norm(OptDataMng.currentGrad);

itr = 1;
while(1)
    
    % Compute nonstatinarity measure
    OptData.Fval(itr) = OptDataMng.currentFval;
    OptData.optimality(itr) = ...
        norm(OptDataMng.currentGrad.*OptDataMng.inactive);
    OptData.actualReduction(itr) = OptDataMng.actualReduction;
    [projControl] = project(Primal.Control, -1, OptDataMng.currentGrad);
    OptDataMng.nonStatinarityMeas = norm(OptDataMng.inactive .* ...
        (Primal.Control.current - projControl));
    OptData.nonStatinarityMeas(itr) = OptDataMng.nonStatinarityMeas;
    % Compute other stopping criteria
    plotResults(Primal.Control.current);
    if(OptDataMng.nonStatinarityMeas <= OptDataMng.stepTol)
        OptData.StoppingCriterion = 'NonstationarityMeasure';
        break;
    elseif(itr > 1 && abs(OptDataMng.actualReduction) <= OptDataMng.actualRedTol)
        OptData.StoppingCriterion = 'ActualReduction';
        break;
    elseif(itr >= OptDataMng.maxOptItr)
        OptData.StoppingCriterion = 'MaxItr';
        break;
    end
    % Compute adaptive constants to ensure superlinear convergence 
    Primal.Control.epsilon = min(1e-3,OptDataMng.nonStatinarityMeas^0.75);
    OptDataMng.eta = 0.1*min(1e-1,OptDataMng.nonStatinarityMeas^0.95);
    % Solve trust region subproblem
    OptDataMng.newTrialControl = false;
    while(1)
        [Primal,MidPrimal,OptDataMng] = ...
            trustRegionSubProblem(Operators,Primal,OptDataMng);
        if(OptDataMng.newTrialControl == true)
            break;
        else
            % Sample more high fidelity data
            Primal.State.highFidelity = true;
            OptDataMng.currentObjFuncInexactness = 0;
            OptDataMng.currentGrad = OptDataMng.currentHighFidelGrad;
            OptDataMng.currentFval = OptDataMng.currentHighFidelFval;
            OptDataMng.trustRegionRadius = OptDataMng.trustRegionRadius * ...
                OptDataMng.trustRegionReduction;
        end
    end
    % Update current primal and gradient information
    [OptDataMng,Primal] = ...
        updateState(Operators,OptDataMng,Primal,MidPrimal);
    % Check gradient inexactness and update model if necessary
    if(itr > 1 && Primal.State.highFidelity == false)
        OptDataMng.numGradientCheck = OptDataMng.numGradientCheck + 1;
        [Primal,OptDataMng] = ...
            checkGradientInexactness(Operators,OptDataMng,Primal);
        OptDataMng.currentObjFuncInexactness = ...
            abs(OptDataMng.currentFval - OptDataMng.currentHighFidelFval);
    else
        % Update objective model
        [Primal.State] = updateModel(Primal.State);
        Primal.State.updateModel = true;
        Primal.State.highFidelity = false;
        OptDataMng.currentObjFuncInexactness = 0;
        OptDataMng.currentHighFidelFval = OptDataMng.currentFval;
        OptDataMng.currentHighFidelGrad = OptDataMng.currentGrad;
    end
    % Update iteration count
    itr = itr + 1;
end
OptData.Itr = itr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,Primal] = ...
    updateState(Operators,OptDataMng,Primal,MidPrimal)

% Compute new midpoint gradient
[OptDataMng.midGrad,~] = gradient(Operators,MidPrimal);
% Evaluate objective function and update control
Primal.Control.old = Primal.Control.current;
[OptDataMng,Primal] = ...
    update(Operators,Primal,MidPrimal.Control,OptDataMng);
% Compute gradient
[OptDataMng.currentGrad,OptDataMng.currentDual] = ...
    gradient(Operators,Primal);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,OptDataMng] = ...
    checkGradientInexactness(Operators,OptDataMng,Primal)

Primal.State.updateModel = false;
% Compute new high fidelity objective function
Primal.State.highFidelity = true;
[OptDataMng.currentHighFidelFval,Primal] = ...
    objective(Operators,Primal);
% Compute new high fidelity gradient
OptDataMng.oldHighFidelGrad = OptDataMng.currentHighFidelGrad;
[OptDataMng.currentHighFidelGrad,OptDataMng.currentHighFidelDual] = ...
    gradient(Operators,Primal);
Primal.State.highFidelity = false;
% Check gradient inexactness
inexactGradientBound = ...
    min(norm(OptDataMng.currentGrad),OptDataMng.trustRegionRadius);
misfit = OptDataMng.currentGrad - OptDataMng.currentHighFidelGrad;
inexactGradientMeasure = norm(misfit);
OptDataMng.inexactGradientMeasure = inexactGradientMeasure;
while(OptDataMng.inexactGradientMeasure > inexactGradientBound)
    % Update gradient model
    [Primal.State] = updateModel(Primal.State);
    [OptDataMng.currentGrad,OptDataMng.currentHighFidelDual] = ...
        gradient(Operators,Primal);
    misfit = OptDataMng.currentGrad - OptDataMng.currentHighFidelGrad;
    OptDataMng.inexactGradientMeasure = norm(misfit);
    Primal.State.updateModel = true;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Variable] = updateModel(Variable)

% Compute eigenvectors and eigenvalues
numSnapshots = Variable.numSnapshots;
A = Variable.storage(:,1:numSnapshots)'*Variable.storage(:,1:numSnapshots);
B = speye(size(A));
% Compute eigenvalues and eigenvectors 
[eigenvectors,D] = eigs(A,B);
eigenvalues = diag(D);
% Gather maximum energy eigenvectors
threshold = 0.9999;
eigenVectorCount = 1;
totalEnergy = sum(eigenvalues);
while(true)
    energy = sum(eigenvalues(1:eigenVectorCount))/totalEnergy;
    if(energy >= threshold)
        break;
    end
    eigenVectorCount = eigenVectorCount + 1;
end
% Compute new orthogonal basis
values = 1 ./ sqrt(eigenvalues);
newBasis = ...
    Variable.storage(:,1:numSnapshots)*eigenvectors(:,1:eigenVectorCount);
for i=1:eigenVectorCount
    newBasis(:,i) = values(i) .* newBasis(:,i);
end
% Compute new orthogonal basis
if(size(Variable.basis,1) > 1)
    Variable.basis = [Variable.basis newBasis];
    Variable.basis = orth(Variable.basis);
else
    Variable.basis = newBasis;
end
Variable.numModelUpdates = Variable.numModelUpdates + 1;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,Primal] = ...
    update(Operators,Primal,MidControl,OptDataMng)

xi = 1;
beta = 1e-2;
mu4 = 1-1e-4;
stop = true;
maxNumItr = 5;

iteration = 1;
MidControl.lowerBound = Primal.Control.lowerBound;
MidControl.upperBound = Primal.Control.upperBound;
while(stop == true)
    lambda = -xi/OptDataMng.alpha;
    % Project new trial point
    [Primal.Control.current] = ...
        project(MidControl,lambda,OptDataMng.midGrad);
    % Compute objective function
    [OptDataMng.currentFval,Primal] = objective(Operators,Primal);
    % Compute actual reduction 
    actualReduction = OptDataMng.currentFval - OptDataMng.midFval;
    if(actualReduction < -mu4*OptDataMng.actualReduction)
        OptDataMng.actualReduction = actualReduction;
        break;
    elseif(iteration >= maxNumItr)
        break;
    end
    % Compute scaling for next iteration
    if(iteration == 1)
        xi = OptDataMng.alpha;
    else
        xi = xi*beta;
    end
    iteration = iteration + 1;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,TrialPrimal,OptDataMng] = ...
    trustRegionSubProblem(Operators,Primal,OptDataMng)

itr = 1;
rflag = false;
maxPcgItr = 200;
TrialPrimal = Primal;
if(OptDataMng.trustRegionRadius < 1e-4)
    OptDataMng.trustRegionRadius = 1e-4;
end

while(itr <= OptDataMng.maxSubProbItr)
    
    % Set Steihaug-Toint CG solver tolerance
    normIGrad = norm(OptDataMng.inactive .* OptDataMng.currentGrad);
    if(normIGrad > 0)
        condition = OptDataMng.trustRegionRadius / (normIGrad + 1e-16);
    else
        condition = OptDataMng.trustRegionRadius / ...
            (norm(OptDataMng.currentGrad) + 1e-16);
    end
    % Compute active and inactive sets
    lambda = min(condition,1);
    [OptDataMng.active,OptDataMng.inactive] = ...
        computeActiveAndInactiveSets(Primal.Control,-lambda,OptDataMng.currentGrad);
    normProjGradient = norm(OptDataMng.inactive.*OptDataMng.currentGrad);
    stoppingTol = OptDataMng.eta*normProjGradient;
    % Compute descent direction
    [descentDirection,~] = ...
        steihaugTointCg(Operators,Primal,OptDataMng,stoppingTol,maxPcgItr);
    % Project trial control
    [TrialPrimal.Control.current] = ...
        project(Primal.Control, 1, descentDirection);
    % Compute predicted reduction
    projTrialStep = TrialPrimal.Control.current - Primal.Control.current;
    [hessTimesProjStep] = ...
        applyVectorToHessian(Operators,OptDataMng,Primal,projTrialStep);
    OptDataMng.predictedReduction = ...
        projTrialStep'*(OptDataMng.inactive.*OptDataMng.currentGrad) + ...
        0.5 * (projTrialStep'*hessTimesProjStep);
    % Check objective function inexactness
    OptDataMng.ObjFuncInexactnessBound = OptDataMng.etaF * ...
        OptDataMng.trustRegionLowerBound * ...
        abs(OptDataMng.predictedReduction);
%     OptDataMng.ObjFuncInexactnessBound = OptDataMng.etaF * ...
%         abs(OptDataMng.predictedReduction);
    if(OptDataMng.currentObjFuncInexactness > ...
            OptDataMng.ObjFuncInexactnessBound)
        OptDataMng.newTrialControl = false;
        break;
    end
    % Evaluate new objective function and equality constraint
    [OptDataMng.midFval,Primal,TrialPrimal] = ...
        evaluateObjective(Operators,Primal,TrialPrimal);
    % Compute actual reduction based on new control and state values
    if(Primal.State.highFidelity == true)
        OptDataMng.actualReduction = ...
            OptDataMng.midFval - OptDataMng.currentHighFidelFval;
    else
        OptDataMng.actualReduction = ...
            OptDataMng.midFval - OptDataMng.currentFval;
    end
    % Compute actual over predicted reduction ratio
    OptDataMng.actualOverPredRed = OptDataMng.actualReduction / ...
        (OptDataMng.predictedReduction + 1e-16);
    % update trust region radius
    [OptDataMng,rflag,stop] = ...
        updateTrustRegionRadius(rflag,Primal.Control,OptDataMng);
    if(stop == true)
        OptDataMng.newTrialControl = true;
        break;
    end
    itr = itr+1;
end

OptDataMng.subProbItrs = itr;
if(itr > OptDataMng.maxSubProbItr)
    OptDataMng.newTrialControl = true;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,rflag,stop] = ...
    updateTrustRegionRadius(rflag,Control,OptDataMng)

mu0 = 1e-4;
stop = false;
inactiveGradient = OptDataMng.inactive .* OptDataMng.currentGrad;
lambda = min(OptDataMng.trustRegionRadius/norm(inactiveGradient),1);
[projControl] = project(Control, -lambda, inactiveGradient);
condition = -OptDataMng.nonStatinarityMeas * mu0 * ...
    norm(Control.current - projControl);

if(OptDataMng.actualReduction >= condition)
    OptDataMng.trustRegionRadius = ...
        OptDataMng.trustRegionReduction * OptDataMng.trustRegionRadius;
    rflag = true;
elseif(OptDataMng.actualOverPredRed < OptDataMng.trustRegionLowerBound)
    OptDataMng.trustRegionRadius = ...
        OptDataMng.trustRegionReduction * OptDataMng.trustRegionRadius;
    rflag = true;
elseif(OptDataMng.actualOverPredRed >= OptDataMng.trustRegionLowerBound && ...
        OptDataMng.actualOverPredRed < OptDataMng.trustRegionMidBound)
    stop = true;
elseif(OptDataMng.actualOverPredRed >= OptDataMng.trustRegionMidBound && ...
        OptDataMng.actualOverPredRed < OptDataMng.trustRegionUpperBound)
    OptDataMng.trustRegionRadius = ...
        OptDataMng.trustRegionExpansion * OptDataMng.trustRegionRadius;
    stop = true;
elseif(OptDataMng.actualOverPredRed > OptDataMng.trustRegionUpperBound && rflag == true)
    OptDataMng.trustRegionRadius = ...
        2 * OptDataMng.trustRegionExpansion * OptDataMng.trustRegionRadius;
    stop = true;
else
    OptDataMng.trustRegionRadius = ...
        OptDataMng.trustRegionExpansion * OptDataMng.trustRegionRadius;
    OptDataMng.trustRegionRadius = ...
        min(OptDataMng.maxTrustRegion,OptDataMng.trustRegionRadius);
end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [descentDirection,cauchyStep] = ...
    steihaugTointCg(Operators,Primal,OptDataMng,tolerance,maxItr)

% initialize newton step
descentDirection = zeros(size(Primal.Control.current));
cauchyStep = descentDirection;
% initialize descent direction
residual = -1 .* (OptDataMng.inactive .* OptDataMng.currentGrad);
% norm residual
normResidual = norm(residual);
% Conjugate direction
conjugateDir = -zeros(size(residual));
% Start Krylov solver
itr = 1;
while(normResidual > tolerance)
    if(itr >= maxItr)
        break;
    end
    % Apply preconditioner
    residualTimesVector = ...
        applyInvPrecOperator(Primal,OptDataMng,residual);
    % compute scaling
    currentTau = residualTimesVector'*residual;
    if(itr == 1)
        conjugateDir = residualTimesVector;
    else
        beta = currentTau / previousTau;
        conjugateDir = residualTimesVector + (beta.*conjugateDir);
    end
    % Apply conjugate direction to hessian operator
    [HessTimesConjugateDir] = ...
        applyVectorToHessian(Operators,OptDataMng,Primal,conjugateDir);
    curvature = conjugateDir'*HessTimesConjugateDir;
    if(curvature <= 0)
        % compute scaled inexact trial step
        scaling = dogleg(Primal,descentDirection,conjugateDir,OptDataMng);
        descentDirection = descentDirection + scaling * conjugateDir;
        break;
    end
    rayleighQuotient = currentTau / curvature;
    residual = residual - rayleighQuotient*HessTimesConjugateDir;
    normResidual = norm(residual);
    descentDirection = descentDirection + rayleighQuotient * conjugateDir;
    if(itr == 1)
        cauchyStep = descentDirection;
    end
    normDescentDirection = norm(descentDirection,2);
    if(normDescentDirection > OptDataMng.trustRegionRadius)
        % compute scaled inexact trial step
        scaling = dogleg(Primal,descentDirection,conjugateDir,OptDataMng);
        descentDirection = descentDirection + scaling * conjugateDir;
        break;
    end
    previousTau = currentTau;
    itr = itr + 1;
end

if(norm(descentDirection) <= 0)
    descentDirection = -1.*(OptDataMng.inactive .* OptDataMng.currentGrad);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [scaling] = ...
    dogleg(Primal,newtonStep,conjugateDir,OptDataMng)

[PrecTimesConjugateDir] = ...
    applyPrecOperator(Primal,OptDataMng,conjugateDir);
stepDotPrecTimesConjugateDir = newtonStep'*PrecTimesConjugateDir;
conjugateDirDotPrecTimesConjugateDir = conjugateDir'*PrecTimesConjugateDir;

[PrecTimesTrialStep] = ...
    applyPrecOperator(Primal,OptDataMng,newtonStep);
trialStepDotPrecTimesTrialStep = newtonStep'*PrecTimesTrialStep;

a = stepDotPrecTimesConjugateDir*stepDotPrecTimesConjugateDir;
b = conjugateDirDotPrecTimesConjugateDir ...
    *(OptDataMng.trustRegionRadius*OptDataMng.trustRegionRadius - ...
    trialStepDotPrecTimesTrialStep);
numerator = -stepDotPrecTimesConjugateDir + sqrt(a + b);
scaling = numerator / conjugateDirDotPrecTimesConjugateDir;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [projControl] = project(Control, alpha, direction)

projControl = Control.current + alpha*direction;
projControl = max(projControl,Control.lowerBound);
projControl = min(projControl,Control.upperBound);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [activeSet,inactiveSet] = ...
    computeActiveAndInactiveSets(Control, alpha, direction)

projControl = Control.current + alpha*direction;
lowerLimit = Control.lowerBound - Control.epsilon;
upperLimit = Control.upperBound + Control.epsilon;
activeSet = ((projControl >= upperLimit) | (projControl <= lowerLimit));
inactiveSet = ~activeSet;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [invPrecTimesVector] = ...
    applyInvPrecOperator(Primal,OptDataMng,vector)

activeSetTimesVector = OptDataMng.active .* vector;
invPrecTimesInactiveVector = OptDataMng.inactive .* vector;
invPrecTimesVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* invPrecTimesInactiveVector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [PrecTimesVector] = ...
    applyPrecOperator(Primal,OptDataMng,vector)

activeSetTimesVector = OptDataMng.active .* vector;
PrecTimesInactiveVector = OptDataMng.inactive .* vector;
PrecTimesVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* PrecTimesInactiveVector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value,Primal] = objective(Operators,Primal)

global GLB_INVP;

Control = Primal.Control;
numLoadCases = size(GLB_INVP.force,2);
currentNumSnapshots = Primal.State.numSnapshots;
% Solve equality constraint (PDE)
if(Primal.State.highFidelity == true)
    Primal.State.HFM = Primal.State.HFM + numLoadCases;
    [Primal.State] = Operators.equality.solve(Primal.State,Control);
    first = currentNumSnapshots+1;
    last = currentNumSnapshots + numLoadCases;
    Primal.State.storage(:,first:last) = Primal.State.current;
    Primal.State.numSnapshots = last;
else
    Primal.State.LFM = Primal.State.LFM + numLoadCases;
    [Primal.State] = Operators.equality.solve(Primal.State,Control);
end
% Compute objective function
value = Operators.objective.evaluate(Primal.State,Control);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value,Primal,TrialPrimal] = ...
    evaluateObjective(Operators,Primal,TrialPrimal)

global GLB_INVP;

Control = TrialPrimal.Control;
numLoadCases = size(GLB_INVP.force,2);
currentNumSnapshots = Primal.State.numSnapshots;
% Solve euqality constraint (PDE)
if(Primal.State.highFidelity == true)
    Primal.State.HFM = Primal.State.HFM + 1;
    [TrialPrimal.State] = ...
        Operators.equality.solve(TrialPrimal.State,Control);
    first = currentNumSnapshots + 1;
    last = currentNumSnapshots + numLoadCases;
    Primal.State.storage(:,first:last) = TrialPrimal.State.current;
    Primal.State.numSnapshots = last;
else
    Primal.State.LFM = Primal.State.LFM + numLoadCases;
    [TrialPrimal.State] = ...
        Operators.equality.solve(TrialPrimal.State,Control);
end
% Compute objective function
value = Operators.objective.evaluate(TrialPrimal.State,Control);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad,Dual] = gradient(Operators,Primal)

Dual = zeros(size(Primal.State));
% Compute reduced gradient for sructural topology optimization problem
grad = ...
    Operators.objective.firstDerivativeWrtControl(Primal.State,Primal.Control);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HessTimesVector] = ...
    applyVectorToHessian(Operators,OptDataMng,Primal,vector)

global GLB_INVP;
AVec = OptDataMng.active .* vector;
IVec = OptDataMng.inactive .* vector;

switch GLB_INVP.HessianType
    case 'GaussNewton'
        [HessTimesIVec] = ...
            GaussNewtonHessian(Operators,Primal,OptDataMng.currentDual,IVec);
    case 'FullHessian'
        [HessTimesIVec] = ...
            FullHessian(Operators,Primal,OptDataMng.currentDual,IVec);
    case 'BFGSHessian'
        [HessTimesIVec] = OptDataMng.BFGSnew*IVec;
end

HessTimesVector = AVec + (OptDataMng.inactive .* HessTimesIVec);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HessTimesVector] = GaussNewtonHessian(Operators,Primal,Dual,vector)

Control = Primal.Control;
[F_zz] = ...
    Operators.objective.secondDerivativeWrtControlControl(Primal.State,Control,vector);
[G_zz] = ...
    Operators.equality.secondDerivativeWrtControlControl(Primal.State,Control,Dual,vector);
HessTimesVector = F_zz + G_zz;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng] = BFGSHessian(OptDataMng,Primal)

OptDataMng.BFGSold = OptDataMng.BFGSnew;
deltaControl = Primal.Control.current - Primal.Control.old;
deltaGradient = OptDataMng.currentHighFidelGrad - ...
    OptDataMng.oldHighFidelGrad;
hessTimesdeltaControl = OptDataMng.BFGSold*deltaControl;
deltaControlTimesHess=deltaControl'*OptDataMng.BFGSold;

alpha = deltaGradient'*deltaControl;
curvature = deltaControl'*(hessTimesdeltaControl);
OptDataMng.BFGSnew = OptDataMng.BFGSold + ...
    (curvature.*(hessTimesdeltaControl*deltaControlTimesHess)) + ...
    (alpha.*(deltaGradient*deltaGradient'));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [error,Primal,Dual] = checkGradient(Operators,Primal)

rng(1);
step = randn(size(Primal.Control.current));
% Compute objective function values
[~,Primal] = objective(Operators,Primal);
% Compute gradient
[grad,Dual] = gradient(Operators,Primal);
trueGradDotStep = grad'*step;

error = zeros(10,1);
TestPrimal = Primal;
for i=1:10
    epsilon = 1/(10^(i-1));
    TestPrimal.Control.current = Primal.Control.current + (epsilon.*step);
    [FvalP,TestPrimal] = objective(Operators,TestPrimal);
    TestPrimal.Control.current = Primal.Control.current - (epsilon.*step);
    [FvalM,TestPrimal] = objective(Operators,TestPrimal);
    finiteDiffAppx = (FvalP - FvalM) / (2*epsilon);
    error(i) = abs(trueGradDotStep-finiteDiffAppx);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [error] = checkHessian(Operators,OptDataMng,Primal)
%
rng(1);
vector = randn(size(Primal.Control.current));
% Compute true Hessian of the Lagrangian
[trueHessian] = ...
    applyVectorToHessian(Operators,OptDataMng,Primal,vector);
normtrueLagranguanHess = norm(trueHessian);
% Compute finite difference approximation
bound = 1e-16;
error = zeros(10,1);
TestPrimal = Primal;
for i=1:10
    epsilon = 1/(10^(i-1));
    %
    TestPrimal.Control.current = ...
        Primal.Control.current + (epsilon.*vector);
    [grad,~] = gradient(Operators,Primal);
    fd_derivative = 8*grad;
    %
    TestPrimal.Control.current = ...
        Primal.Control.current - (epsilon.*vector);
    [grad,~] = gradient(Operators,Primal);
    fd_derivative = fd_derivative - (8*grad);
    %
    TestPrimal.Control.current = ...
        Primal.Control.current + ((2*epsilon).*vector);
    [grad,~] = gradient(Operators,Primal);
    fd_derivative = fd_derivative - (grad);
    %
    TestPrimal.Control.current = ...
        Primal.Control.current - ((2*epsilon).*vector);
    [grad,~] = gradient(Operators,Primal);
    fd_derivative = fd_derivative + (grad);
    %
    alpha = 1 / (12*epsilon);
    fd_derivative = alpha.*fd_derivative;
    %
    diff = fd_derivative - trueHessian;
    error(i) = norm(diff) / (bound + normtrueLagranguanHess);
end

end