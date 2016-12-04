function [Primal,Results] = ...
    prototype2(gradTol,feasTol,stepTol,stagTol,maxOptItr,maxSubProbItr)

% Problem interface directories
addpath ./interface/;
addpath /Users/maguilo/Research/intrelab;
addpath /Users/maguilo/Research/femlab/tools/;
addpath /Users/maguilo/Research/intrelab/mesh2/;

fprintf('\n*** Elastosatics Topology Optimization ***\n');

global GLB_INVP;

% Domain specifications
Domain.xmin = 0;  % min dim in x-dir
Domain.xmax = 1.6;   % max dim in x-dir
Domain.ymin = 0;  % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 32;     % num intervals in x-dir
Domain.ny = 20;     % num intervals in y-dir
% Regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV';
beta = 0.5*(Domain.xmax / (Domain.nx));
gamma = 1e-1;
% Random number seed
rng(0,'v5normal');
% Right hand force function
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
% TOpology optimization Hessian formulation
GLB_INVP.ProblemType = 'GaussNewton';
% Enable derivative check
derivativeCheck = false;
% Set Primal Space Dimensions
OptDataMng = [];
numConstraints = 1;
OptDataMng.nSlacks = 0;
OptDataMng.nControls = GLB_INVP.nVertGrid;
% Initialize primal vector
Primal = [];
Primal.penalty = 0.1;
Primal.LagMult = 5*ones(numConstraints,1);
Primal.nStates = GLB_INVP.nVertGrid * GLB_INVP.spaceDim;
Primal.State.current = zeros(Primal.nStates,1);
Primal.current = ones(OptDataMng.nControls + OptDataMng.nSlacks,1);
Primal.current(1+OptDataMng.nControls:end) = 0;
% Set primal bounds
Primal.upperBound = ones(OptDataMng.nControls + OptDataMng.nSlacks,1);
if(OptDataMng.nSlacks>0)
    Primal.upperBound(1+OptDataMng.nControls:end) = 1e10;
end
Primal.lowerBound = zeros(OptDataMng.nControls + OptDataMng.nSlacks,1);
% Get Operators interface
Operators = [];
Operators.equality = equalityConstraint;
Operators.objective = objectiveFunction;
Operators.inequality = inequalityConstraint;
% Set topology optimization specific parameters
GLB_INVP.SimpPenalty = 3;
Control = Primal.current(1:OptDataMng.nControls);
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*Control);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
% Objective function normalization factor
ControlData.current = ones(size(Primal.current));
[State] = Operators.equality.solve(Primal.State,ControlData);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = 1 / (State.current'*K*State.current);
clear K State ControlData;
% Set initial guess for control parameters
Primal.current(1:OptDataMng.nControls) = ...
    GLB_INVP.VolumeFraction * ones(size(Control));
% Optimization parameters
OptDataMng.alpha = 1e-2;
OptDataMng.gradTol = gradTol;
OptDataMng.feasTol = feasTol;
OptDataMng.stepTol = stepTol;
OptDataMng.stagTol = stagTol;
OptDataMng.alphaPenalty = 0.1;
OptDataMng.gammaAugLagGrad = 1e0;
OptDataMng.minPenaltyValue = 1e-8;
OptDataMng.maxOptItr = maxOptItr;
OptDataMng.maxSubProbItr = maxSubProbItr;
% Trust region parameters
OptDataMng.maxTrustRegion = 1e3;
OptDataMng.trustRegionExpansion = 2;
OptDataMng.trustRegionMidBound = 0.25;
OptDataMng.trustRegionReduction = 0.25;
OptDataMng.trustRegionLowerBound = 0.1;
OptDataMng.trustRegionUpperBound = 0.75;
% Construct active and inactive sets
OptDataMng.active = zeros(size(Primal.current));
OptDataMng.inactive = ones(size(Primal.current));
% Solve problem
proctime = cputime;
walltime = tic;
if(derivativeCheck == true)
    [Results.gradError,OptDataMng,Primal] = ...
        checkGradient(Operators,OptDataMng,Primal);
    [Results.hessError] = checkHessian(Operators,OptDataMng,Primal);
else
    [Primal,Results] = getMin(Operators,Primal,OptDataMng);
end
proctime = cputime - proctime;
walltime = toc(walltime);
% Print CPU and wall clock times to screen
fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', walltime);
% Plot optimal control
plotResults(-1*Primal.current(1:OptDataMng.nControls));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,OptData] = getMin(Operators,Primal,OptDataMng)

% Compute objective function
[OptDataMng.currentFval,OptDataMng.currentHval,...
    OptDataMng.currentlval,OptDataMng.currentLval,Primal] = ...
    evaluateAugLag(Operators,OptDataMng,Primal);
OptDataMng.actualReduction = OptDataMng.currentLval;
% Compute augmented Lagrangian gradient
[OptDataMng.currentFgrad,OptDataMng.currentHgrad,...
    OptDataMng.currentlgrad,OptDataMng.currentLgrad] = ...
    computeAugLagGrad(Operators,OptDataMng,Primal);
OptDataMng.stagMes = norm(Primal.current);
OptDataMng.normHval = norm(OptDataMng.currentHval);
OptDataMng.normlgrad = norm(OptDataMng.currentlgrad);
OptDataMng.normLgrad = norm(OptDataMng.currentLgrad);
% Solve optimization problem
OptData.Fval = zeros(OptDataMng.maxOptItr,1);
OptData.Hval = zeros(OptDataMng.maxOptItr,1);
OptData.lval = zeros(OptDataMng.maxOptItr,1);
OptData.Lval = zeros(OptDataMng.maxOptItr,1);
OptData.Hnorm = zeros(OptDataMng.maxOptItr,1);
OptData.lnorm = zeros(OptDataMng.maxOptItr,1);
OptData.StagMeas = zeros(OptDataMng.maxOptItr,1);
OptData.nonStatinarityMeas = zeros(OptDataMng.maxOptItr,1);
OptDataMng.trustRegionRadius = norm(OptDataMng.currentLgrad,2);

itr = 1;
while(1)
    
    % Save optimization output history
    [OptData] = saveOptData(itr,OptData,OptDataMng);
    % Compute nonstatinarity measure
    [projControl] = project(Primal, -1, OptDataMng.currentLgrad);
    TrialStep = Primal.current - projControl;
    OptDataMng.nonStatinarityMeas = norm(OptDataMng.inactive .* TrialStep);
    OptData.nonStatinarityMeas(itr) = OptDataMng.nonStatinarityMeas;
    % Compute adaptive constants to ensure superlinear convergence
    Primal.epsilon = min(1e-3,OptDataMng.nonStatinarityMeas^0.75);
    OptDataMng.eta = 0.1*min(1e-1,OptDataMng.nonStatinarityMeas^0.95);
    % Solve trust region subproblem
    [Primal,MidPrimal,OptDataMng] = ...
        trustRegionSubProblem(Operators,Primal,OptDataMng);
    % Compute new midpoint gradients
    [OptDataMng.midFgrad,OptDataMng.midHgrad,...
        OptDataMng.midlgrad,OptDataMng.midLgrad] = ...
        computeAugLagGrad(Operators,OptDataMng,Primal);
    % Update primal
    OptDataMng.OldPrimal = Primal;
    [OptDataMng,Primal] = ...
        updatePrimal(Operators,Primal,MidPrimal,OptDataMng);
    % Compute new gradients
    [OptDataMng.currentFgrad,OptDataMng.currentHgrad,...
        OptDataMng.currentlgrad,OptDataMng.currentLgrad] = ...
        computeAugLagGrad(Operators,OptDataMng,Primal);
    % Compute stopping criteria metrics
    nz = OptDataMng.nControls;
    OptDataMng.normHval = norm(OptDataMng.currentHval);
    OptDataMng.normlgrad = ...
        norm(OptDataMng.inactive(1:nz).*OptDataMng.currentlgrad);
    OptDataMng.normLgrad = ...
        norm(OptDataMng.inactive(1:nz).*OptDataMng.currentLgrad(1:nz));
    DeltaControl = OptDataMng.OldPrimal.current - Primal.current;
    OptDataMng.stagMes = norm(OptDataMng.inactive .* DeltaControl);
    % Check stopping criteria
    condition = OptDataMng.gammaAugLagGrad * Primal.penalty;
    %plotResults(-1*Primal.current(1:OptDataMng.nControls));
    if(OptDataMng.normLgrad <= condition)
        [stop,why] = ...
            checkStoppingCriteria(itr,OptDataMng);
        if(stop == true)
            OptData.why = why;
            % Save final output data
            [OptData] = saveOptData(itr+1,OptData,OptDataMng);
            break;
        else
            % Update Lagrange multipliers and penalty parameter
            [stop,why,Primal] = ...
                updateLagrangeMultipliers(OptDataMng,Primal);
            if(stop == true)
                OptData.why = why;
                % Save final output data
                [OptData] = saveOptData(itr+1,OptData,OptDataMng);
                break;
            end
        end
    else
        % Check stopping criteria
        if(itr >= OptDataMng.maxOptItr)
            OptData.why = 'MaxItr';
            break;
        elseif(OptDataMng.normHval <= OptDataMng.feasTol && ...
                OptDataMng.normlgrad <= OptDataMng.gradTol)
            OptData.why = 'FeasibilityAndOptimalityMet';
            % Save final output data
            [OptData] = saveOptData(itr+1,OptData,OptDataMng);
            break;
        end
    end
    % Update iteration count
    itr = itr + 1;
    
end
OptData.OuterItr = itr;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptData] = saveOptData(itr,OptData,OptDataMng)

% Save optimization output history
OptData.Hnorm(itr) = OptDataMng.normHval;
OptData.lnorm(itr) = OptDataMng.normlgrad;
OptData.Fval(itr) = OptDataMng.currentFval;
OptData.Hval(itr) = OptDataMng.currentHval;
OptData.lval(itr) = OptDataMng.currentlval;
OptData.Lval(itr) = OptDataMng.currentLval;
OptData.StagMeas(itr) = OptDataMng.stagMes;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [stop,why,Primal] = ...
    updateLagrangeMultipliers(OptDataMng,Primal)

stop = false;
why = 'NotConverged';
currentPenalty = Primal.penalty;
Primal.penalty = OptDataMng.alphaPenalty*Primal.penalty;
if(Primal.penalty >= OptDataMng.minPenaltyValue)
    Primal.LagMult = Primal.LagMult + ...
        ((1/currentPenalty)*OptDataMng.currentHval);
else
    stop = true;
    why = 'PenaltyTol';
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [stop,why] = checkStoppingCriteria(outerItr,OptDataMng)

stop = false;
why = 'NotConverged';
% Check stopping criteria
if(outerItr >= OptDataMng.maxOptItr)
    stop = true;
    why = 'MaxItr';
elseif(OptDataMng.normHval <= OptDataMng.feasTol && ...
        OptDataMng.normlgrad <= OptDataMng.gradTol)
    stop = true;
    why = 'FeasibilityAndOptimalityMet';
elseif(OptDataMng.stagMes <= OptDataMng.stagTol)
    stop = true;
    why = 'Stagnation';
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,Primal] = ...
    updatePrimal(Operators,Primal,MidPrimal,OptDataMng)

xi = 1;
beta = 1e-2;
mu4 = 1-1e-4;
stop = true;
maxNumItr = 10;

iteration = 1;
MidPrimal.lowerBound = Primal.lowerBound;
MidPrimal.upperBound = Primal.upperBound;
while(stop == true)
    lambda = -xi/OptDataMng.alpha;
    % Project new trial point
    [Primal.current] = project(MidPrimal,lambda,OptDataMng.midLgrad);
    % Compute objective function
    [OptDataMng.currentFval,OptDataMng.currentHval,...
        OptDataMng.currentlval,OptDataMng.currentLval,Primal] = ...
        evaluateAugLag(Operators,OptDataMng,Primal);
    % Compute actual reduction
    actualReduction = OptDataMng.currentLval - OptDataMng.midLval;
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

TrialPrimal.LagMult = Primal.LagMult;
TrialPrimal.penalty = Primal.penalty;
TrialPrimal.current = zeros(size(Primal.current));
TrialPrimal.State.current = zeros(size(Primal.State.current));

if(OptDataMng.trustRegionRadius < 1e-4)
    OptDataMng.trustRegionRadius = 1e-4;
end

itr = 1;
rflag = false;
maxPcgItr = 200;
while(itr <= OptDataMng.maxSubProbItr)
    
    % Compute condition for direction step length
    normIGrad = norm(OptDataMng.inactive .* OptDataMng.currentLgrad);
    if(normIGrad > 0)
        condition = OptDataMng.trustRegionRadius / normIGrad;
    else
        condition = OptDataMng.trustRegionRadius / ...
            norm(OptDataMng.currentGrad);
    end
    % Update active and inactive sets
    lambda = min(condition,1);
    [OptDataMng.active,OptDataMng.inactive] = ...
        computeActiveAndInactiveSets(Primal,-lambda,OptDataMng.currentLgrad);
    % Set solver stopping tolerance
    normProjGradient = norm(OptDataMng.inactive.*OptDataMng.currentLgrad);
    stoppingTol = OptDataMng.eta*normProjGradient;
    % Compute descent direction
    [descentDirection,~] = ...
        steihaugTointCg(Operators,Primal,OptDataMng,stoppingTol,maxPcgItr);
    % Project trial control
    [TrialPrimal.current] = project(Primal, 1, descentDirection);
    % Evaluate midpoint objective function and residual
    [OptDataMng.midFval,OptDataMng.midHval,...
        OptDataMng.midlval,OptDataMng.midLval,TrialPrimal] = ...
        evaluateAugLag(Operators,OptDataMng,TrialPrimal);
    % Compute actual reduction based on trial primal
    OptDataMng.actualReduction = ...
        OptDataMng.midLval - OptDataMng.currentLval;
    % Compute predicted reduction
    projTrialStep = TrialPrimal.current - Primal.current;
    [augLagHessTimesVec] = ...
        computeAugLagHess(Operators,OptDataMng,Primal,projTrialStep);
    predictedReduction = ...
        projTrialStep'*(OptDataMng.inactive.*OptDataMng.currentLgrad) + ...
        0.5 * (projTrialStep'*augLagHessTimesVec);
    % Compute actual over predicted reduction ratio
    OptDataMng.actualOverPredRed = ...
        OptDataMng.actualReduction / predictedReduction;
    % update trust region radius
    [OptDataMng,rflag,stop] = ...
        updateTrustRegionRadius(rflag,Primal,OptDataMng);
    if(stop == true)
        break;
    end
    itr = itr+1;
end

OptDataMng.subProbItrs = itr;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,rflag,stop] = ...
    updateTrustRegionRadius(rflag,Primal,OptDataMng)

mu0 = 1e-4;
stop = false;
inactiveGradient = OptDataMng.inactive .* OptDataMng.currentLgrad;
lambda = min(OptDataMng.trustRegionRadius/norm(inactiveGradient),1);
[projControl] = project(Primal, -lambda, inactiveGradient);
condition = -OptDataMng.nonStatinarityMeas*mu0*norm(Primal.current - projControl);

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
    stop = true;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [descentDirection, cauchyStep] = ...
    steihaugTointCg(Operators,Primal,OptDataMng,tolerance,maxItr)
% initialize newton step
cauchyStep = zeros(size(Primal.current));
descentDirection = zeros(size(Primal.current));
% initialize descent direction
residual = -1 .* (OptDataMng.inactive .* OptDataMng.currentLgrad);
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
    [augLagHessTimesConjugateDir] = ...
        computeAugLagHess(Operators,OptDataMng,Primal,conjugateDir);
    curvature = conjugateDir'*augLagHessTimesConjugateDir;
    if(curvature <= 0)
        % compute scaled inexact trial step
        scaling = dogleg(Primal,descentDirection,conjugateDir,OptDataMng);
        descentDirection = descentDirection + scaling * conjugateDir;
        break;
    end
    rayleighQuotient = currentTau / curvature;
    residual = residual - rayleighQuotient*augLagHessTimesConjugateDir;
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
    descentDirection = cauchyStep;
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

function [projControl] = project(Primal, alpha, direction)

projControl = Primal.current + alpha*direction;
projControl = max(projControl,Primal.lowerBound);
projControl = min(projControl,Primal.upperBound);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [activeSet,inactiveSet] = ...
    computeActiveAndInactiveSets(Primal, alpha, direction)

OptDataMng.nControls = size(Primal.current,1);
activeSet = zeros(size(Primal.current));
inactiveSet = zeros(size(Primal.current));
projControl = Primal.current + alpha*direction;
for i=1:OptDataMng.nControls
    lowerLimit = Primal.lowerBound(i) - Primal.epsilon;
    upperLimit = Primal.upperBound(i) + Primal.epsilon;
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

function [Fval,Hval,lval,Lval,Primal] = ...
    evaluateAugLag(Operators,OptDataMng,Primal)

ns = OptDataMng.nSlacks;
nz = OptDataMng.nControls;
Control.current = Primal.current(1:nz);
% Evaluate partial differential equation
[Primal.State] = Operators.equality.solve(Primal.State,Control);
% Evaluate objective funtion 
Fval = Operators.objective.evaluate(Primal.State,Control);
% Evaluate inequality constraint
Hval = Operators.inequality.residual(Primal.State,Control);
if(ns>0)
    Slacks = Primal.current(1+nz:end);
    Hval = Hval - Slacks;
end
% Evaluate Lagrangian
lval = Fval + (Primal.LagMult'*Hval);
% Evaluate augmented Lagrangian
Lval = lval + ((0.5/Primal.penalty)*(Hval'*Hval));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Fgrad,Hgrad,lgrad,Lgrad] = ...
    computeAugLagGrad(Operators,OptDataMng,Primal)

ns = OptDataMng.nSlacks;
nz = OptDataMng.nControls;
Lgrad = zeros(nz+ns,1);
Control.current = Primal.current(1:nz);
% Compute objective function gradient
Fgrad = ...
    Operators.objective.firstDerivativeWrtControl(Primal.State,Control);
% Compute inequality constraint Jacobian (i.e. gradient)
Hgrad = ...
    Operators.inequality.firstDerivativeWrtControl(Primal.State,Control);
% Compute Lagrangian gradient
lgrad = Fgrad + (Hgrad'*Primal.LagMult);
% Compute augmented Lagrangian gradient
Lgrad(1:nz) = lgrad + ((1/Primal.penalty)*(Hgrad'*OptDataMng.currentHval));
if(ns>0)
    % Compute gradient for slack variables
    Lgrad(1+nz:end) = ...
        -(Primal.LagMult + ((1/Primal.penalty)*OptDataMng.currentHval));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lagHessTimesVec] = ...
    computeLagrangianHess(Operators,OptDataMng,Primal,vector)

nz = OptDataMng.nControls;

dz = vector(1:nz);
State = Primal.State;
Control.current = Primal.current(1:nz);
% compute objective contribution
F_hess = ...
    Operators.objective.secondDerivativeWrtControlControl(State,Control,dz);
% apply perturbation to inequality constraint Hessian
%   **** Multiple constraints will require a loop ****
penalty = 1/Primal.penalty;
H_zz_times_dz = ...
    Operators.inequality.secondDerivativeWrtControlControl(State,Control,dz);
H_hess = H_zz_times_dz * Primal.LagMult + ...
    (penalty .* H_zz_times_dz * OptDataMng.currentHval);
% Compute Lagrangian Hessian
lagHessTimesVec = F_hess + H_hess;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [augLagHessTimesVec] = ...
    computeAugLagHess(Operators,OptDataMng,Primal,vector)

activeVecSet = OptDataMng.active .* vector;
inactiveVecSet = OptDataMng.inactive .* vector;
% Compute Lagrangian Hessian
[lagHessTimesIVec] = ...
    computeLagrangianHess(Operators,OptDataMng,Primal,vector);
% Apply vector to constraint Jacobian
nz = OptDataMng.nControls;
Hgrad_x_Ivec = OptDataMng.currentHgrad*inactiveVecSet(1:nz);
HgradTHgrad_x_IVec = OptDataMng.currentHgrad' * Hgrad_x_Ivec;
% Compute augmented Lagrangian Hessian
ns = OptDataMng.nSlacks;
penalty = (1/Primal.penalty);
augLagHessTimesVec = zeros(nz+ns,1);
if(ns > 0)
    ds = vector(1+nz:end);
    augLagHessTimesVec(1:nz) = activeVecSet(1:nz) + ...
        (OptDataMng.inactive(1:nz).*lagHessTimesIVec) + ...
        (OptDataMng.inactive(1:nz).*(penalty.*HgradTHgrad_x_IVec)) - ...
        (OptDataMng.inactive(1:nz).*(penalty.*(OptDataMng.currentHgrad'*ds)));
    augLagHessTimesVec(1+nz:end) = activeVecSet(1+nz:end) + ...
        (OptDataMng.inactive(1+nz:end) .* (ds - (penalty.*Hgrad_x_Ivec)));
else
    augLagHessTimesVec(1:nz) = activeVecSet + ...
        (OptDataMng.inactive .* lagHessTimesIVec) + ...
        (OptDataMng.inactive .* (penalty .* HgradTHgrad_x_IVec));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [error,OptDataMng,Primal] = ...
    checkGradient(Operators,OptDataMng,Primal)

rng(1);
step = randn(size(Primal.current));
TestPrimal.State = Primal.State;
TestPrimal.LagMult = Primal.LagMult;
TestPrimal.penalty = Primal.penalty;
TestPrimal.current = Primal.current;

% Compute objective function and inequality constraints values
[OptDataMng.currentFval,OptDataMng.currentHval,...
    OptDataMng.currentlval,OptDataMng.currentLval,Primal] = ...
    evaluateAugLag(Operators,OptDataMng,Primal);
% Compute gradients
[OptDataMng.currentFgrad,OptDataMng.currentHgrad,...
    OptDataMng.currentlgrad,OptDataMng.currentLgrad] = ...
    computeAugLagGrad(Operators,OptDataMng,Primal);
trueGradDotStep = OptDataMng.currentLgrad'*step;

error = zeros(10,1);
for i=1:10
    epsilon = 1/(10^(i-1));
    TestPrimal.current = Primal.current + (epsilon.*step);
    [~,~,~,LvalP,TestPrimal] = ...
        evaluateAugLag(Operators,OptDataMng,TestPrimal);
    TestPrimal.current = Primal.current - (epsilon.*step);
    [~,~,~,LvalM,TestPrimal] = ...
        evaluateAugLag(Operators,OptDataMng,TestPrimal);
    finiteDiffAppx = (LvalP - LvalM) / (2*epsilon);
    error(i) = abs(trueGradDotStep-finiteDiffAppx);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [error] = checkHessian(Operators,OptDataMng,Primal)
%
rng(1);
vector = randn(size(Primal.current));
TestPrimal.State = Primal.State;
TestPrimal.LagMult = Primal.LagMult;
TestPrimal.penalty = Primal.penalty;
TestPrimal.current = Primal.current;
% Compute true Hessian of the Lagrangian
[trueLagrangianHess] = ...
    computeLagrangianHess(Operators,OptDataMng,Primal,vector);
normtrueLagranguanHess = norm(trueLagrangianHess);
% Compute finite difference approximation
bound = 1e-16;
error = zeros(10,1);
for i=1:10
    epsilon = 1/(10^(i-1));
    %
    TestPrimal.current = Primal.current + (epsilon.*vector);
    [~,~,~,Lgrad] = computeAugLagGrad(Operators,OptDataMng,TestPrimal);
    fd_derivative = 8*Lgrad;
    %
    TestPrimal.current = Primal.current - (epsilon.*vector);
    [~,~,~,Lgrad] = computeAugLagGrad(Operators,OptDataMng,TestPrimal);
    fd_derivative = fd_derivative - (8*Lgrad);
    %
    TestPrimal.current = Primal.current + ((2*epsilon).*vector);
    [~,~,~,Lgrad] = computeAugLagGrad(Operators,OptDataMng,TestPrimal);
    fd_derivative = fd_derivative - (Lgrad);
    %
    TestPrimal.current = Primal.current - ((2*epsilon).*vector);
    [~,~,~,Lgrad] = computeAugLagGrad(Operators,OptDataMng,TestPrimal);
    fd_derivative = fd_derivative + (Lgrad);
    %
    alpha = 1 / (12*epsilon);
    fd_derivative = alpha.*fd_derivative;
    %
    diff = fd_derivative - trueLagrangianHess;
    error(i) = norm(diff) / (bound + normtrueLagranguanHess);
end

end