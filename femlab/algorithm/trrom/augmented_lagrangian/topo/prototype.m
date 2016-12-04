function [Primal,data] = ...
    prototype(stepTol,feasTol,actualRedTol,maxOptItr,maxSubProbItr)

% Prototype for surrogate based topology optimization
% NOTES:
% - Based on C.T. Kelley and E.W. Sachs 1999 paper: a trust region method
% for parabolic boundary control problems
clc;

% Problem interface directories
addpath ./interface/;
addpath ./assemblyInqLP/;
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
Domain.nx = 32;     % num intervals in x-dir
Domain.ny = 20;     % num intervals in y-dir
% Regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV';
beta = 0.5*(Domain.xmax / (Domain.nx));
gamma = 1e-1;
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
GLB_INVP.ProblemType = 'LP';

% Set Primal Space Dimensions
OptDataMng = [];
numConstraints = 1;
OptDataMng.nSlacks = numConstraints;
OptDataMng.nControls = GLB_INVP.nVertGrid;
% Initialize primal vector
Primal = [];
Primal.penalty = 1;
Primal.LagMult = ones(numConstraints,1);
Primal.current = ones(GLB_INVP.nVertGrid + numConstraints,1);
Primal.current(1+OptDataMng.nControls:end) = 0;
% Set primal bounds
Primal.upperBound = ones(size(Primal.current));
Primal.upperBound(1+OptDataMng.nControls:end) = inf;
Primal.lowerBound = 1e-3*ones(size(Primal.current));
Primal.lowerBound(1+OptDataMng.nControls:end) = 0;

% Get Operators interface
Operators = [];
Operators.assemble = assemblyRoutines;
Operators.equality = equalityConstraint;
Operators.objective = objectiveFunction;
Operators.inequality = inequalityConstraint;

% Store Topology Optimization Problem Specific Parameters
Control = Primal.current(1:OptDataMng.nControls);
GLB_INVP.SimpPenalty = 3;
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*Control);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
% Normalization factors
Data.current = 0.5*ones(size(Primal.current));
State.numStates = GLB_INVP.nVertGrid * GLB_INVP.spaceDim;
[State] = Operators.equality.solve(State,Data);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = 0.5 / (State.current'*K*State.current);
clear K;

% Solve reduced order model based optimization problem
OptDataMng.alpha = 1e-2;
OptDataMng.alphaEta = 0.9;
OptDataMng.feasTol = feasTol;
OptDataMng.stepTol = stepTol;
OptDataMng.alphaPenalty = 3;
OptDataMng.augLagGradTol = 1e-1;
OptDataMng.maxOptItr = maxOptItr;
OptDataMng.relativeFeasTol = feasTol;
OptDataMng.actualRedTol = actualRedTol;
OptDataMng.maxSubProbItr = maxSubProbItr;
% Trust region parameters
OptDataMng.maxTrustRegion = 1e3;
OptDataMng.trustRegionExpansion = 2;
OptDataMng.trustRegionMidBound = 0.25;
OptDataMng.trustRegionReduction = 0.25;
OptDataMng.trustRegionLowerBound = 0.1;
OptDataMng.trustRegionUpperBound = 0.75;
% Set control initial guess
Primal.current(1:OptDataMng.nControls) = ...
    GLB_INVP.VolumeFraction * ones(size(Control));

proctime = cputime;
walltime = tic;
[Primal,data] = getMin(Operators,Primal,State,OptDataMng);
proctime = cputime - proctime;
walltime = toc(walltime);

fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', walltime);

% Plot Results
 plotResults(-1*Primal.current(1:OptDataMng.nControls));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Primal,OptData] = getMin(Operators,Primal,State,OptDataMng)

% Construct initial inactive set
OptDataMng.inactive = ones(size(Primal));
% Evaluate initial augmented Lagrangian function
[OptDataMng.currentLval,OptDataMng.currentFval,OptDataMng.currentHval,State] = ...
    evaluateAugLag(Operators,OptDataMng,State,Primal);
OptDataMng.actualReduction = OptDataMng.currentLval;
% Assemble reduced gradient
[OptDataMng.currentGrad,Dual] = ...
    computeAugLagGrad(Operators,OptDataMng,State,Primal);
%
TestPrimal.LagMult = Primal.LagMult;
TestPrimal.penalty = Primal.penalty;
TestPrimal.current = Primal.current;
p = randn(size(Primal.current));
error = zeros(10,1);
TST = OptDataMng.currentGrad'*p;
for i=1:10
    epsilon = 1/(10^(i-1));
    TestPrimal.current = Primal.current + (epsilon.*p);
    [FdLvalP,FdFval,FdHval,State] = ...
        evaluateAugLag(Operators,OptDataMng,State,TestPrimal);
    TestPrimal.current = Primal.current - (epsilon.*p);
    [FdLvalM,FdFval,FdHval,State] = ...
        evaluateAugLag(Operators,OptDataMng,State,TestPrimal);
    FD = (FdLvalP - FdLvalM) / (2*epsilon);
    error(i) = abs(TST-FD);
end
% Solve optimization problem
OptData.Lval = zeros(OptDataMng.maxOptItr,1);
OptData.Hval = zeros(OptDataMng.maxOptItr,1);
OptData.Fval = zeros(OptDataMng.maxOptItr,1);
OptData.nonStationarityMeas = zeros(OptDataMng.maxOptItr,1);
OptDataMng.trustRegionRadius = norm(OptDataMng.currentGrad,2);

itr = 1;
while(1)
    
    % Compute nonstatinarity measure
    OptData.Lval(itr) = OptDataMng.currentLval;
    OptData.Hval(itr) = OptDataMng.currentHval;
    OptData.Fval(itr) = OptDataMng.currentFval;
    [projControl] = project(Primal, -1, OptDataMng.currentGrad);
    OptDataMng.nonStationarityMeas = norm(OptDataMng.inactive .* ...
        (Primal.current - projControl));
    OptData.nonStationarityMeas(itr) = OptDataMng.nonStationarityMeas;
    % Compute other stopping criteria
    plotResults(Primal.current(1:OptDataMng.nControls));
    if(itr > 1 && abs(OptDataMng.actualReduction) <= OptDataMng.actualRedTol)
        OptData.StoppingCriterion = 'ActualReduction';
        break;
    elseif(itr >= OptDataMng.maxOptItr)
        OptData.StoppingCriterion = 'MaxItr';
        break;
    end
    
    % Compute adaptive constants to ensure superlinear convergence 
    Primal.Epsilon = min(1e-3,OptDataMng.nonStationarityMeas^0.75);
    OptDataMng.eta = 0.1*min(1e-1,OptDataMng.nonStationarityMeas^0.95);
    % Solve trust region subproblem
    [MidState,MidPrimal,OptDataMng] = ...
        trustRegionSubProblem(Operators,OptDataMng,State,Primal,Dual);
    % Compute new midpoint gradient
    [OptDataMng.midGrad,~] = ...
        computeAugLagGrad(Operators,OptDataMng,MidState,MidPrimal);
    % Update state
    [OptDataMng,State,Primal] = ...
        update(Operators,OptDataMng,State,Primal,MidPrimal);
    % Compute new gradient
    [OptDataMng.currentGrad,Dual] = ...
        computeAugLagGrad(Operators,OptDataMng,State,Primal);
    % Check feasibility and optimality stopping criteria
    if(OptDataMng.normHval <= OptDataMng.relativeFeasTol)
        if(OptDataMng.normHval <= OptDataMng.feasTol && ...
                OptDataMng.nonStationarityMeas <= OptDataMng.stepTol)
                    OptData.StoppingCriterion = 'FeasibilityAndOptimalityMeas';
            break;
        end
        Primal.LagMult = Primal.LagMult + (Primal.penalty*OptDataMng.currentHval);
        OptDataMng.relativeFeasTol = OptDataMng.relativeFeasTol^OptDataMng.alphaEta;
        OptDataMng.augLagGradTol = OptDataMng.augLagGradTol / Primal.penalty;
    else
        Primal.penalty = OptDataMng.alphaPenalty * Primal.penalty;
        OptDataMng.relativeFeasTol = OptDataMng.relativeFeasTol^(1-OptDataMng.alphaEta);
        OptDataMng.augLagGradTol = 1 / Primal.penalty;
    end
    % Update iteration count
    itr = itr + 1;
end
OptData.Itr = itr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [OptDataMng,State,Primal] = ...
    update(Operators,OptDataMng,State,Primal,MidPrimal)

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
    [Primal.current] = project(MidPrimal,lambda,OptDataMng.midGrad);
        % Evaluate augmented Lagrangian function
    [OptDataMng.currentLval,OptDataMng.currentFval,OptDataMng.currentHval,State] = ...
        evaluateAugLag(Operators,OptDataMng,State,Primal);
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
OptDataMng.normHval = norm(OptDataMng.currentHval);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TrialState,TrialPrimal,OptDataMng] = ...
    trustRegionSubProblem(Operators,OptDataMng,State,Primal,Dual)

itr = 1;
rflag = false;
maxPcgItr = 200;
TrialPrimal.penalty = Primal.penalty;
TrialPrimal.LagMult = Primal.LagMult;
TrialState.current = zeros(size(State.current));
TrialPrimal.current = zeros(size(Primal.current));
    normProjGradient = norm(OptDataMng.inactive.*OptDataMng.currentGrad);
while(itr <= OptDataMng.maxSubProbItr)
    
    % Set solver tolerance
    condition = OptDataMng.trustRegionRadius / ...
        (norm(OptDataMng.inactive .* OptDataMng.currentGrad) + 1e-16);
    lambda = min(condition,1);
    [OptDataMng.active,OptDataMng.inactive] = ...
        computeActiveAndInactiveSets(Primal,-lambda,OptDataMng.currentGrad);
    stoppingTol = OptDataMng.eta*normProjGradient;
    % Compute descent direction
    [descentDirection,~] = ...
        steihaugTointCg(Operators,OptDataMng,State,Primal,Dual,stoppingTol,maxPcgItr);
    % Project trial control
    [TrialPrimal.current] = project(Primal, 1, descentDirection);
    % Evaluate augmented Lagrangian function
    [OptDataMng.midLval,OptDataMng.midFval,OptDataMng.midHval,TrialState] = ...
        evaluateAugLag(Operators,OptDataMng,TrialState,TrialPrimal);
    % Compute actual reduction based on new control and state values
    OptDataMng.actualReduction = OptDataMng.midLval - OptDataMng.currentLval;
    % Compute predicted reduction
    projTrialStep = TrialPrimal.current - Primal.current;
    [kktTimesProjStep] = ...
        applyVectorToKKT(Operators,OptDataMng,State,Primal,Dual,projTrialStep);
    predictedReduction = ...
        projTrialStep'*(OptDataMng.inactive.*OptDataMng.currentGrad) + ...
        0.5 * (projTrialStep'*kktTimesProjStep);
    % Compute actual over predicted reduction ratio
    OptDataMng.actualOverPredRed = OptDataMng.actualReduction / ...
        (predictedReduction + 1e-16);
    % update trust region radius
    [OptDataMng,rflag,stop] = updateTrustRegionRadius(rflag,Primal,OptDataMng);
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
inactiveGradient = OptDataMng.inactive .* OptDataMng.currentGrad;
lambda = min(OptDataMng.trustRegionRadius/norm(inactiveGradient),1);
[projControl] = project(Primal, -lambda, inactiveGradient);
condition = -OptDataMng.nonStationarityMeas*mu0*norm(Primal.current - projControl);

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
    steihaugTointCg(Operators,OptDataMng,State,Primal,Dual,tolerance,maxItr)
% initialize newton step
descentDirection = zeros(size(Primal.current));
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
        applyInvPrecOperator(OptDataMng,State,Primal,Dual,residual);
    % compute scaling
    currentTau = residualTimesVector'*residual;
    if(itr == 1)
        conjugateDir = residualTimesVector;
    else
        beta = currentTau / previousTau;
        conjugateDir = residualTimesVector + (beta.*conjugateDir);
    end
    % Apply conjugate direction to hessian operator
    [kktTimesConjugateDir] = ...
        applyVectorToKKT(Operators,OptDataMng,State,Primal,Dual,conjugateDir);
    curvature = conjugateDir'*kktTimesConjugateDir;
    if(curvature <= 0)
        % compute scaled inexact trial step
        scaling = ...
            dogleg(OptDataMng,State,Primal,Dual,descentDirection,conjugateDir);
        descentDirection = descentDirection + scaling * conjugateDir;
        break;
    end
    rayleighQuotient = currentTau / curvature;
    residual = residual - rayleighQuotient*kktTimesConjugateDir;
    normResidual = norm(residual);
    descentDirection = descentDirection + rayleighQuotient * conjugateDir;
    if(itr == 1)
        cauchyStep = descentDirection;
    end
    normDescentDirection = norm(descentDirection,2);
    if(normDescentDirection > OptDataMng.trustRegionRadius)
        % compute scaled inexact trial step
        scaling = ...
            dogleg(OptDataMng,State,Primal,Dual,descentDirection,conjugateDir);
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
    dogleg(OptDataMng,State,Primal,Dual,newtonStep,conjugateDir)

[PrecTimesConjugateDir] = ...
    applyPrecOperator(OptDataMng,State,Primal,Dual,conjugateDir);
stepDotPrecTimesConjugateDir = newtonStep'*PrecTimesConjugateDir;
conjugateDirDotPrecTimesConjugateDir = conjugateDir'*PrecTimesConjugateDir;

[PrecTimesTrialStep] = ...
    applyPrecOperator(OptDataMng,State,Primal,Dual,newtonStep);
trialStepDotPrecTimesTrialStep = newtonStep'*PrecTimesTrialStep;

a = stepDotPrecTimesConjugateDir*stepDotPrecTimesConjugateDir;
b = conjugateDirDotPrecTimesConjugateDir * ...
    (OptDataMng.trustRegionRadius*OptDataMng.trustRegionRadius - ...
    trialStepDotPrecTimesTrialStep);
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

% Compute control active and inactive sets
projControl = Primal.current + alpha*direction;
lowerLimit = Primal.lowerBound - Primal.Epsilon;
upperLimit = Primal.upperBound + Primal.Epsilon;
activeSet = ((projControl >= upperLimit) | (projControl <= lowerLimit));
inactiveSet = ~activeSet;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [invPrecTimesVector] = ...
    applyInvPrecOperator(OptDataMng,State,Primal,Dual,vector)

activeSetTimesVector = OptDataMng.active .* vector;
invPrecTimesInactiveVector = OptDataMng.inactive .* vector;
invPrecTimesVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* invPrecTimesInactiveVector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [PrecTimesVector] = ...
    applyPrecOperator(OptDataMng,State,Primal,Dual,vector)

activeSetTimesVector = OptDataMng.active .* vector;
PrecTimesInactiveVector = OptDataMng.inactive .* vector;
PrecTimesVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* PrecTimesInactiveVector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Lval,Fval,Hval,State] = ...
    evaluateAugLag(Operators,OptDataMng,State,Primal)

% Evaluate objective function
[Fval,State] = ...
    Operators.assemble.objective(Operators,OptDataMng,State,Primal);
% Evaluate inequality constraint
[Hval] = Operators.assemble.inequality(Operators,OptDataMng,State,Primal);
% Evaluate augmented Lagrangian
%Lval = Fval + (Primal.LagMult' * Hval) + (0.5 * Primal.penalty * (Hval' * Hval));
Lval = Fval;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [augLagGrad,Dual] = ...
    computeAugLagGrad(Operators,OptDataMng,State,Primal)
% KKT system
%
%              | L_x |
%  \nabla{L} = | L_s |
%              | L_l | 
%
Dual = [];
ns = OptDataMng.nSlacks;
nz = OptDataMng.nControls;
augLagGrad = zeros(nz+ns,1);
% Compute objective constribution to augmented Lagrangian gradient
[Fgrad,Dual.objective] = ...
   Operators.assemble.objectiveGrad(Operators,OptDataMng,State,Primal);
% Compute constraint contributions to augmented Lagrangian gradient 
%               **** Multiple constraints require loop ****
%Lambda = Primal.LagMult + (Primal.penalty .* OptDataMng.currentHval);
[Hgrad,Dual.inequality] = ...
    Operators.assemble.inequalityGrad(Operators,OptDataMng,State,Primal);
%augLagGrad(1:nz) = Fgrad + (Hgrad' .* Lambda);
% Evaluate inequality constraint
%augLagGrad(1:nz) = augLagGrad(1:nz) + Hgrad;
augLagGrad(1:nz) = Fgrad;
augLagGrad(1+nz:end) = 0;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [KKTxVector] = ...
    applyVectorToKKT(Operators,OptDataMng,State,Primal,Dual,vector)
% KKT system
%
%               |   L_xx     -mu*Jac^T  Jac^T |
% \nabla^2{L} = | -mu*Jac        1       -1   |
%               |   Jac         -1        0   | 
%

activeSetTimesVector = OptDataMng.active .* vector;
inactiveSetTimesVector = OptDataMng.inactive .* vector;
[HessTimesInactiveVector] = ...
    Operators.assemble.hessian(Operators,OptDataMng,State,Primal,Dual,inactiveSetTimesVector);
KKTxVector = activeSetTimesVector + ...
    (OptDataMng.inactive .* HessTimesInactiveVector);

end
