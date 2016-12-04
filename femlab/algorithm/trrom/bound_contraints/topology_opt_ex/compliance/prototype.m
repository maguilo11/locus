function [Primal,Results] = ...
    prototype(stepTol,actualRedTol,maxOptItr,maxSubProbItr)

% Prototype for surrogate based topology optimization
% NOTES:
% - Based on C.T. Kelley and E.W. Sachs 1999 paper: a trust region method
% for parabolic boundary control problems
clc;

% Problem interface directories
addpath ./interface;
addpath /Users/miguelaguilo/dotk/matlab/exe/;
addpath /Users/miguelaguilo/Research/intrelab;
addpath /Users/miguelaguilo/Research/femlab/tools/;
addpath /Users/miguelaguilo/Research/intrelab/mesh2/;

global GLB_INVP;

fprintf('\n*** Elastosatics Topology Optimization ***\n');

% Domain specifications
Domain.xmin = 0;  % min dim in x-dir
Domain.xmax = 1.6;   % max dim in x-dir
Domain.ymin = 0;  % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 96;     % num intervals in x-dir
Domain.ny = 60;     % num intervals in y-dir
% Regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV';
gamma = 1e-1;
beta = (Domain.xmax / (Domain.nx));
% the regularization parameter beta is related to the level of noise
% the larger the noise, the larger beta should be
rng(0,'v5normal');

rhs_fn=@(usr_par)generateNodalForce(usr_par);

% generate problem-specific constant quantities
[GLB_INVP] = generateParams(Domain, rhs_fn);

% store problem-specific constant quantites
GLB_INVP.reg = reg;
GLB_INVP.beta = beta;
GLB_INVP.gamma = gamma;
GLB_INVP.model_t = 'simp';
GLB_INVP.VolumeFraction = 0.3;
% Helmholtz ilter radius of influence
GLB_INVP.epsilon = 1/1e3;
% Formulation
GLB_INVP.ProblemType = 'GaussNewton';

% Set Control Space Dimension and Initial Guess
Primal.Control.numControls = GLB_INVP.nVertGrid;
Primal.Control.current = ones(GLB_INVP.nVertGrid,1);
Primal.Control.upperBound = ones(size(Primal.Control.current));
Primal.Control.lowerBound = zeros(size(Primal.Control.current));

Operators = [];
Operators.equality = equalityConstraint;
Operators.objective = objectiveFunction;
Operators.inequality = inequalityConstraint;
% Store Topology Optimization Problem Specific Parameters
GLB_INVP.SimpPenalty = 3;
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*Primal.Control.current);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
% Normalization factors
Primal.State.numStates = GLB_INVP.nVertGrid * GLB_INVP.spaceDim;
GLB_INVP.alpha = 1e0 / (sum(GLB_INVP.Ms*(GLB_INVP.VolumeFraction*Primal.Control.current))^2);
[Primal.State] = Operators.equality.solve(Primal.State,Primal.Control);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = 0.5 / (Primal.State.current'*K*Primal.State.current);
clear K;

% Solve reduced order model based optimization problem
OptDataMng.alpha = 1e-2;
OptDataMng.stepTol = stepTol;
OptDataMng.maxOptItr = maxOptItr;
OptDataMng.actualRedTol = actualRedTol;
OptDataMng.maxSubProbItr = maxSubProbItr;
% Trust region parameters
OptDataMng.maxTrustRegion = 1e3;
OptDataMng.trustRegionExpansion = 2;
OptDataMng.trustRegionMidBound = 0.25;
OptDataMng.trustRegionReduction = 0.25;
OptDataMng.trustRegionLowerBound = 0.1;
OptDataMng.trustRegionUpperBound = 0.75;
% State and Control data initialization
Primal.Control.current = ...
    GLB_INVP.VolumeFraction * ones(Primal.Control.numControls,1);

proctime = cputime;
walltime = tic;
[Primal,Results] = getMin(Operators,Primal,OptDataMng);
proctime = cputime - proctime;
walltime = toc(walltime);

fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', walltime);

% Plot Results
 plotResults(-1*Primal.Control.current);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,OptData] = getMin(Operators,Primal,OptDataMng)
% Compute objective function
[OptDataMng.currentFval,Primal.State] = ...
    objective(Operators,Primal.State,Primal.Control);
OptDataMng.actualReduction = OptDataMng.currentFval;
% Compute Lagrange multipliers
rhs = 0;
[OptDataMng.currentDual] = ...
        Operators.equality.applyInverseAdjointJacobianWrtState(Primal.State,Primal.Control,rhs);
% Construct State Sotrage
OptDataMng.inactive = ones(size(Primal.Control.current));
% Assemble reduced gradient
[OptDataMng.currentGrad] =  Operators.objective.firstDerivativeWrtControl(Primal.State,Primal.Control);
% Solve optimization problem
OptData.Fval = zeros(OptDataMng.maxOptItr,1);
OptData.nonStatinarityMeas = zeros(OptDataMng.maxOptItr,1);
OptData.deltaObjectiveFunction = zeros(OptDataMng.maxOptItr,1);
OptDataMng.trustRegionRadius = norm(OptDataMng.currentGrad,2);

itr = 1;
while(1)
    
    % Compute nonstatinarity measure
    OptData.Fval(itr) = OptDataMng.currentFval;
    [projControl] = project(Primal.Control, -1, OptDataMng.currentGrad);
    OptDataMng.nonStatinarityMeas = norm(OptDataMng.inactive .* (Primal.Control.current - projControl));
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
    [Primal,MidState,MidControl,OptDataMng] = ...
        trustRegionSubProblem(Operators,Primal,OptDataMng);
    % Compute new midpoint gradient
    [OptDataMng.midGrad] = Operators.objective.firstDerivativeWrtControl(MidState,MidControl);
    % Update state
    [OptDataMng,Primal] = ...
        update(Operators,Primal,MidControl,OptDataMng);
    % Compute new gradient
    [OptDataMng.currentGrad] = Operators.objective.firstDerivativeWrtControl(Primal.State,Primal.Control);
    % Update iteration count
    itr = itr + 1;
end
OptData.Itr = itr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,Primal] = ...
    update(Operators,Primal,MidControl,OptDataMng)

xi = 1;
beta = 1e-2;
mu4 = 1-1e-4;
stop = true;
maxNumItr = 10;

iteration = 1;
MidControl.lowerBound = Primal.Control.lowerBound;
MidControl.upperBound = Primal.Control.upperBound;
while(stop == true)
    lambda = -xi/OptDataMng.alpha;
    % Project new trial point
    [Primal.Control.current] = project(MidControl,lambda,OptDataMng.midGrad);
    % Compute objective function
    [OptDataMng.currentFval,Primal.State] = ...
        objective(Operators,Primal.State,Primal.Control);
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

function [Primal,TrialState,TrialControl,OptDataMng] = ...
    trustRegionSubProblem(Operators,Primal,OptDataMng)

TrialState.current = zeros(size(Primal.State.current));
TrialControl.current = zeros(size(Primal.Control.current));

itr = 1;
rflag = false;
maxPcgItr = 200;
if(OptDataMng.trustRegionRadius < 1e-4)
    OptDataMng.trustRegionRadius = 1e-4;
end

while(itr <= OptDataMng.maxSubProbItr)
    
    % Set solver tolerance
    condition = OptDataMng.trustRegionRadius / ...
        (norm(OptDataMng.inactive .* OptDataMng.currentGrad) + 1e-16);
    lambda = min(condition,1);
    [OptDataMng.active,OptDataMng.inactive] = ...
        computeActiveAndInactiveSets(Primal.Control,-lambda,OptDataMng.currentGrad);
    normProjGradient = norm(OptDataMng.inactive.*OptDataMng.currentGrad);
    stoppingTol = OptDataMng.eta*normProjGradient;
    % Compute descent direction
    [descentDirection] = steihaugTointCg(Operators,Primal,OptDataMng,stoppingTol,maxPcgItr);
    % Project trial control
    [TrialControl.current] = project(Primal.Control, 1, descentDirection);
    % Evaluate new objective function and residual
    [OptDataMng.midFval,TrialState] = objective(Operators,TrialState,TrialControl);
    % Compute actual reduction based on new control and state values
    OptDataMng.actualReduction = OptDataMng.midFval - OptDataMng.currentFval;
    % Compute predicted reduction
    projTrialStep = TrialControl.current - Primal.Control.current;
    [hessTimesProjStep] = ...
        applyVectorToHessian(Operators,Primal,OptDataMng,projTrialStep);
    predictedReduction = ...
        projTrialStep'*(OptDataMng.inactive.*OptDataMng.currentGrad) + ...
        0.5 * (projTrialStep'*hessTimesProjStep);
    % Compute actual over predicted reduction ratio
    OptDataMng.actualOverPredRed = OptDataMng.actualReduction / ...
        (predictedReduction + 1e-16);
    % update trust region radius
    [OptDataMng,rflag,stop] = ...
        updateTrustRegionRadius(rflag,Primal.Control,OptDataMng);
    if(stop == true)
        break;
    end
    itr = itr+1;
end

OptDataMng.subProbItrs = itr;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,rflag,stop] = ...
    updateTrustRegionRadius(rflag,Control,OptDataMng)

mu0 = 1e-4;
stop = false;
inactiveGradient = OptDataMng.inactive .* OptDataMng.currentGrad;
lambda = min(OptDataMng.trustRegionRadius/norm(inactiveGradient),1);
[projControl] = project(Control, -lambda, inactiveGradient);
condition = -OptDataMng.nonStatinarityMeas*mu0*norm(Control.current - projControl);

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

function [descentDirection, cauchyStep] = ...
    steihaugTointCg(Operators,Primal,OptDataMng,tolerance,maxItr)
% initialize newton step
descentDirection = zeros(size(Primal.Control.current));
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
    [HessTimesConjugateDir] = applyVectorToHessian(Operators,Primal,OptDataMng,conjugateDir);
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
    descentDirection = ...
        -1 .* (OptDataMng.inactive .* OptDataMng.currentGrad);
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
    *(OptDataMng.trustRegionRadius*OptDataMng.trustRegionRadius - trialStepDotPrecTimesTrialStep);
numerator = -stepDotPrecTimesConjugateDir + sqrt(a + b);
scaling = numerator / conjugateDirDotPrecTimesConjugateDir;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [projControl] = project(Control, alpha, direction)

projControl = Control.current + alpha*direction;
numControls = size(projControl,1);
for i=1:numControls
    projControl(i) = max(projControl(i),Control.lowerBound(i));
    projControl(i) = min(projControl(i),Control.upperBound(i));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [activeSet,inactiveSet] = ...
    computeActiveAndInactiveSets(Control, alpha, direction)

activeSet = zeros(Control.numControls,1);
inactiveSet = zeros(Control.numControls,1);
projControl = Control.current + alpha*direction;
for i=1:Control.numControls
    lowerLimit = Control.lowerBound(i) - Control.epsilon;
    upperLimit = Control.upperBound(i) + Control.epsilon;
    activeSet(i) = ((projControl(i) >= upperLimit) || (projControl(i) <= lowerLimit));
    inactiveSet(i) = ~activeSet(i);
end

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

function [HessTimesVector] = ...
    applyVectorToHessian(Operators,Primal,OptDataMng,vector)

HessTimesVector = applyHessOperatorHFM(Operators,Primal,OptDataMng,vector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HessTimesVector] = ...
    applyHessOperatorHFM(Operators,Primal,OptDataMng,vector)

global GLB_INVP;
activeSetTimesVector = OptDataMng.active .* vector;
inactiveSetTimesVector = OptDataMng.inactive .* vector;

switch GLB_INVP.ProblemType
    case 'GaussNewton'
        [HessTimesInactiveVector] = ...
            highFidelityHessTimesVecTypeLP(Operators,Primal,inactiveSetTimesVector);
    case 'FullNewton'
        [HessTimesInactiveVector] = ...
            highFidelityHessTImesVecTypeNLP(Operators,Primal,inactiveSetTimesVector);
end

HessTimesVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* HessTimesInactiveVector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HessTimesVector] = highFidelityHessTimesVecTypeLP(Operators,Primal,vector)
[objective_component] =  Operators.objective.secondDerivativeWrtControlControl(Primal.State,Primal.Control,vector);
Dual = zeros(size(Primal.State.current));
[equality_component] =  Operators.equality. secondDerivativeWrtControlControl(Primal.State,Primal.Control,Dual,vector);
HessTimesVector = objective_component + equality_component;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value,State] = objective(Operators,State,Control)
% Solve euqality constraint (PDE)
[State] = Operators.equality.solve(State,Control);
% Compute objective function
value = Operators.objective.evaluate(State,Control);
end
