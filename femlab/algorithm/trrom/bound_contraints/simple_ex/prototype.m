function [Primal,Results] = ...
    prototype(stepTol,actualRedTol,maxOptItr,maxSubProbItr)

% Prototype for surrogate based topology optimization
% NOTES:
% - Based on C.T. Kelley and E.W. Sachs 1999 paper: a trust region method
% for parabolic boundary control problems
clc;

% Problem interface directories
addpath ./interface/;
addpath /Users/miguelaguilo/Research/intrelab;

fprintf('\n*** Optimization - Rosenbrock ***\n');

% Set Control Space Dimension and Initial Guess
Primal.Control.numControls = 2;
Primal.Control.current = ones(Primal.Control.numControls,1);
Primal.Control.upperBound = 1e3*ones(size(Primal.Control.current));
Primal.Control.lowerBound = -1e3*ones(size(Primal.Control.current));
% Set operators
Operators = [];
Operators.objective = objectiveFunction;
% Solve reduced order model based optimization problem
OptDataMng.alpha = 1e-2;
OptDataMng.stepTol = stepTol;
OptDataMng.maxOptItr = maxOptItr;
OptDataMng.actualRedTol = actualRedTol;
OptDataMng.maxSubProbItr = maxSubProbItr;
% Trust region parameters
OptDataMng.maxTrustRegion = 1e6;
OptDataMng.trustRegionExpansion = 2;
OptDataMng.trustRegionMidBound = 0.25;
OptDataMng.trustRegionReduction = 0.5;
OptDataMng.trustRegionLowerBound = 0.1;
OptDataMng.trustRegionUpperBound = 0.75;
% State and Control data initialization
Primal.Control.current = 2 * ones(Primal.Control.numControls,1);

proctime = cputime;
walltime = tic;
[Primal,Results,OptDataMng] = getMin(Operators,Primal,OptDataMng);
proctime = cputime - proctime;
walltime = toc(walltime);

fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', walltime);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,OptData,OptDataMng] = getMin(Operators,Primal,OptDataMng)
% Compute objective function
[OptDataMng.currentFval] = objective(Operators,Primal.Control);
OptDataMng.actualReduction = OptDataMng.currentFval;
% Construct inactive set
OptDataMng.inactive = ones(size(Primal.Control.current));
% Assemble reduced gradient
[OptDataMng.currentGrad] = ...
    Operators.objective.firstDerivative(Primal.Control.current);
% Solve optimization problem
OptData.Fval = zeros(OptDataMng.maxOptItr,1);
OptData.nonStatinarityMeas = zeros(OptDataMng.maxOptItr,1);
OptData.actualReduction = zeros(OptDataMng.maxOptItr,1);
OptDataMng.trustRegionRadius = norm(OptDataMng.currentGrad,2);

itr = 1;
while(1)
    
    % Compute nonstatinarity measure
    OptData.Fval(itr) = OptDataMng.currentFval;
    OptData.actualReduction(itr) = OptDataMng.actualReduction;
    [projControl] = project(Primal.Control, -1, OptDataMng.currentGrad);
    OptDataMng.nonStatinarityMeas = ...
        norm(OptDataMng.inactive .* (Primal.Control.current - projControl));
    OptData.nonStatinarityMeas(itr) = OptDataMng.nonStatinarityMeas;
    % Compute stopping criteria
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
    [Primal,MidControl,OptDataMng] = ...
        trustRegionSubProblem(Operators,Primal,OptDataMng);
    % Compute new midpoint gradient
    [OptDataMng.midGrad] = ...
        Operators.objective.firstDerivative(MidControl.current);
    % Update state
    [OptDataMng,Primal] = update(Operators,Primal,MidControl,OptDataMng);
    % Compute new gradient
    [OptDataMng.currentGrad] = ...
        Operators.objective.firstDerivative(Primal.Control.current);
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
    [OptDataMng.currentFval] = objective(Operators,Primal.Control);
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

function [Primal,TrialControl,OptDataMng] = ...
    trustRegionSubProblem(Operators,Primal,OptDataMng)

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
    [descentDirection,~] = ...
        steihaugTointCg(Operators,Primal,OptDataMng,stoppingTol,maxPcgItr);
    % Project trial control
    [TrialControl.current] = project(Primal.Control, 1, descentDirection);
    % Evaluate new objective function and residual
    [OptDataMng.midFval] = objective(Operators,TrialControl);
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
    OptDataMng.actualOverPredRed = ...
        OptDataMng.actualReduction / (predictedReduction + 1e-16);
    % update trust region radius
    [OptDataMng,rflag,stop] = ...
        updateTrustRegionRadius(rflag,Primal.Control,OptDataMng);
    if(stop == true)
        break;
    end
    itr = itr+1;
end

OptDataMng.subProbItrs = itr-1;

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
    [HessTimesConjugateDir] = ...
        applyVectorToHessian(Operators,Primal,OptDataMng,conjugateDir);
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
projControl = max(projControl,Control.lowerBound);
projControl = min(projControl,Control.upperBound);

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

activeSetTimesVector = OptDataMng.active .* vector;
inactiveSetTimesVector = OptDataMng.inactive .* vector;

[HessTimesInactiveVector] = ...
    hessTimesVec(Operators,Primal,inactiveSetTimesVector);

HessTimesVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* HessTimesInactiveVector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HessTimesVector] = hessTimesVec(Operators,Primal,vector)
[HessTimesVector] = ...
    Operators.objective.secondDerivative(Primal.Control.current,vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = objective(Operators,Control)
% Compute objective function
value = Operators.objective.evaluate(Control.current);
end
