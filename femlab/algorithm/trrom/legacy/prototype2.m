function [solution,data] = ...
    prototype2(deltaControlTol,gradTol,stepTol,ObjFuncTol,maxOptItr,maxSubProbItr,basisTreshold)
% Prototype for reduced model based topology optimization
clc;

% Problem interface directories
addpath ./interface2/;
addpath /Users/maguilo/Research/intrelab;
addpath /Users/maguilo/Research/femlab/tools/;
addpath /Users/maguilo/Research/intrelab/mesh2/;

global GLB_INVP;

fprintf('\n*** Structural Topology Optimization ***\n');

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
% Normalization factors
GLB_INVP.theta = 1;
GLB_INVP.alpha = 1;
% Formulation
GLB_INVP.ProblemType = 'LP';

% Set Control Space Dimension and Initial Guess
NumberControls = GLB_INVP.nVertGrid;
InitialControl = GLB_INVP.VolumeFraction * ones(NumberControls,1);

% Get objective function operators
objective = objectiveFunction;
% Get equality constraint operators
equality = equalityConstraint;

% Store Topology Optimization Problem Specific Parameters
GLB_INVP.SimpPenalty = 3;
one = ones(GLB_INVP.nVertGrid,1);
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*one);
GLB_INVP.alpha = 1 / (sum(GLB_INVP.Ms*(0.1*one))^2);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
[state] = equality.solve(one);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = 0.5 / (state'*K*state);
clear K;

proctime = cputime;
walltime = tic;
% Solve reduced order model based optimization problem
[solution,data] = ...
    getMin(objective,equality,InitialControl,deltaControlTol,gradTol,stepTol,ObjFuncTol,maxOptItr,maxSubProbItr,basisTreshold);
proctime = cputime - proctime;
walltime = toc(walltime);

fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', walltime);

% Plot Results
 plotResults(-1*solution);
end

function [control,Data] = ...
    getMin(objective,equality,control,deltaControlTol,gradTol,stepTol,ObjFuncTol,maxItr,maxSubProbItr,basisTreshold)
% Solve pde constraint
[oldState] = equality.solve(control);
% Compute objective function
oldFval = objective.evaluate(oldState,control);
% Construct State Sotrage
maxBasisSize = 100000;
trustRegionLowerBound = -0.2;
numState = size(oldState,1);
stateStorage = zeros(numState,maxBasisSize);
numBasisVectors = 1;
stateStorage(:,1) = oldState;
% Build initial orthogonal basis
stateBasis=0;
% Solve reduced pde constraint
stateReducedBasisCoeff = 0;
% Assemble reduced gradient
[oldGrad] =  objective.firstDerivativeWrtControl(oldState,control);
% Solve optimization problem
itr = 1;
num_eigen_solves = 0;
HFM = true;
update = true;
oldTrustRegion = norm(oldGrad,2);
Data.Fval = zeros(maxItr,1);
Data.deltaControl = zeros(maxItr,1);
Data.normProjTrailStep = zeros(maxItr,1);
Data.normProjCauchyStep = zeros(maxItr,1);
Data.deltaObjectiveFunction = zeros(maxItr,1);
while(1)
    [newControl,newFval,newTrustRegion,projCauchyStep,projTrialStep,numBasisVectors,stateStorage,rho] = ...
        trustRegionSubProblem(objective,equality,stateStorage,stateReducedBasisCoeff,stateBasis,...
        control,oldFval,oldGrad,oldTrustRegion,maxSubProbItr,numBasisVectors,HFM);
    % Check convergence
    Data.Fval(itr) = newFval;
    Data.deltaControl(itr) = max(abs(control-newControl));
    Data.deltaObjectiveFunction(itr) = abs(oldFval-newFval);
    Data.normProjTrailStep(itr) = norm(projTrialStep,2);
    Data.normProjCauchyStep(itr) = norm(projCauchyStep,2);
    plotResults(newControl);
    if(Data.deltaControl(itr) <= deltaControlTol)
        Data.StoppingCriterion = 'DeltaControl';
        break;
    elseif(Data.normProjTrailStep(itr) <= stepTol)
        Data.StoppingCriterion = 'NormProjTrialStep';
        break;
    elseif(Data.deltaObjectiveFunction(itr) <= ObjFuncTol)
        Data.StoppingCriterion = 'DeltaObjFunc';
        break;
    elseif(Data.normProjCauchyStep(itr) <= gradTol)
        Data.StoppingCriterion = 'NormProjCauchyStep';
        break;
    elseif(itr >= maxItr)
        Data.StoppingCriterion = 'MaxItr';
        break;
    end
    %
    if(numBasisVectors > maxBasisSize)
        HFM = false;
    end
    %
    if(itr < maxBasisSize)
        [newGrad] = objective.firstDerivativeWrtControl(stateStorage(:,numBasisVectors),newControl);
    elseif(HFM == false && rho < trustRegionLowerBound)
        % Solve pde constraint
        [state] = equality.solve(newControl);
        stateStorage(:,numBasisVectors + 1) = state;
        numBasisVectors = numBasisVectors+1;
        % update orthogonal basis
        [newBasis] = buildOrthogonalBasis(stateStorage(:,1:numBasisVectors),basisTreshold);
        stateBasis = [stateBasis newBasis];
        stateBasis = orth(stateBasis);
        % Solve reduced pde constraint
        [stateReducedBasisCoeff] = equality.lowFidelitySolve(stateBasis,newControl);
        %[newGrad] = objective.lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,newControl);
        [newGrad] = objective.firstDerivativeWrtControl(stateStorage(:,numBasisVectors),newControl);
        %plotResults(newControl);
        num_eigen_solves = num_eigen_solves + 1;
    elseif(HFM == false && update ==true)
        % update orthogonal basis
        [stateBasis] = buildOrthogonalBasis(stateStorage(:,1:numBasisVectors),basisTreshold);
        % Solve reduced pde constraint
        [stateReducedBasisCoeff] = equality.lowFidelitySolve(stateBasis,newControl);
        % Compute new gradient
        [newGrad] = objective.lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,newControl);
        num_eigen_solves = num_eigen_solves + 1;
        update = false;
    elseif(HFM == false && update ==false)
        [newGrad] = objective.lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,newControl);
    end
    itr = itr + 1;
    %if(newFval < oldFval)
        oldFval = newFval;
        oldGrad = newGrad;
        control = newControl;
        oldTrustRegion = newTrustRegion;
    %end
end
Data.Itr = itr;
Data.NumHfmEval = size(stateStorage,2);
Data.NumEigenSolves = num_eigen_solves;
end

function [basis] = buildOrthogonalBasis(snapShots,threshold)
% Compute eigenvectors and eigenvalues
A = snapShots'*snapShots;
B = speye(size(A));
%N = size(A,2)-1;
[eigenvectors,D] = eigs(A,B);
eigenvalues = diag(D);
% Gather eigenvector set to store based on the total energy
totalEigenValueSum = sum(eigenvalues);
numberBasisFuncs = 1;
while(true)
    energy = sum(eigenvalues(1:numberBasisFuncs))/totalEigenValueSum;
    if(energy >= threshold)
        break;
    end
    numberBasisFuncs = numberBasisFuncs + 1;
end
% Compute orthogonal basis
basis=snapShots*eigenvectors(:,1:numberBasisFuncs);
for i=1:numberBasisFuncs
    const = 1 / sqrt(eigenvalues(i));
    basis(:,i) = const .* basis(:,i);
end

end

function [newControl,newFval,trustRegion,projCauchyStep,projTrialStep,numBasisVectors,stateStorage,rho] = ...
    trustRegionSubProblem(objective,equality,stateStorage,stateReducedBasisCoeff,stateBasis,oldControl,...
    oldFval,oldGradient,trustRegion,maxItr,numBasisVectors,HFM)
itr = 1;
maxPcgItr = 200;
maxProjectionItr = 10;
oldState = stateStorage(:,numBasisVectors);
controlUpperBound = ones(size(oldControl));
controlLowerBound = 1e-2*ones(size(oldControl));
tolerance = 1e-1*norm(oldGradient);
trustRegionReduction = 0.5;
trustRegionExpansion = 2.;
trustRegionLowerBound = 0.2;
trustRegionUpperBound = 0.8;
while(itr <= maxItr)
    % Compute projected Cauchy step
    [projCauchyStep,predRedBasedOnCauchyStep,activeSet] = ...
        projectedCauchyStep(objective,equality,oldState,stateReducedBasisCoeff,stateBasis, ...
        oldControl,controlLowerBound,controlUpperBound,...
        oldGradient,trustRegion,maxProjectionItr,HFM);
    % Compute newton step
    [scaledNewtonStep] = ...
        steihaugTointCg(objective,equality,oldState,stateReducedBasisCoeff,stateBasis,oldControl,...
        oldGradient,trustRegion,tolerance,maxPcgItr,HFM);
    % Update control
    [newControl,predictedReduction,projTrialStep] = ...
        updateControl(objective,equality,oldState,stateReducedBasisCoeff,stateBasis,....
        oldControl,controlLowerBound,controlUpperBound,oldGradient,...
        scaledNewtonStep,predRedBasedOnCauchyStep,trustRegion,...
        maxProjectionItr,HFM);
    if(HFM == true)
        % Solve pde constraint
        [state] = equality.solve(newControl);
        stateStorage(:,numBasisVectors + 1) = state;
        numBasisVectors = numBasisVectors+1;
        % Compute objective function
        [newFval] = objective.evaluate(state,newControl);
    else
        % Low fidelity solve (Old version had a bug, this solve was not done)
        %[stateReducedBasisCoeff] = equality.lowFidelitySolve(stateBasis,newControl);
        % Compute objective function
        [newFval] = objective.lowFidelityEvaluate(stateReducedBasisCoeff,stateBasis,newControl);
        % Compute actual reduction based on new control and state values
    end
    % Compute actual reduction based on new control and state values
    actualReduction = newFval - oldFval;
    % Compute actual over predicted reduction ratio
    rho = actualReduction / predictedReduction;
    if(rho >= trustRegionUpperBound)
        trustRegion = trustRegionExpansion * trustRegion;
        break;
    elseif(rho >= trustRegionLowerBound)
        break;
    else
        trustRegion = trustRegionReduction * trustRegion;
    end
    itr = itr+1;
end
projTrialStep = activeSet .* projTrialStep;
projCauchyStep = activeSet .* projCauchyStep;
end

% function [scaledNewtonStep, newtonStep, cauchyStep] = ...
%     steihaugTointCg(objective,equality,oldState,stateReducedBasisCoeff,stateBasis,...
%     oldControl,gradient,trustRegion,tolerance,maxItr,activeSet,HFM)
function [scaledNewtonStep, newtonStep, cauchyStep] = ...
    steihaugTointCg(objective,equality,oldState,stateReducedBasisCoeff,stateBasis,...
    oldControl,gradient,trustRegion,tolerance,maxItr,HFM)
% initialize newton step
newtonStep = zeros(size(oldControl));
% initialize descent direction
%oldDescentDir = activeSet .* gradient;
oldDescentDir = gradient;
% Apply preconditioner
oldInvPrecTimesVector = ...
    applyInvPrecOperator(stateReducedBasisCoeff,stateBasis,oldControl,oldDescentDir);
% Conjugate direction
conjugateDir = -oldInvPrecTimesVector;
%conjugateDir = activeSet .* conjugateDir;
% Start Krylov solver
itr = 1;
while(1)
    if(itr >= maxItr)
        scaledNewtonStep = newtonStep;
        break;
    end
    % Apply direction to Hessian operator
    if(HFM == true)
        HessTimesConjugateDir = applyHessOperatorHFM(objective,equality,oldState,oldControl,conjugateDir);
    else
        HessTimesConjugateDir = ...
            applyHessOperatorLFM(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,conjugateDir);
    end
    curvature = conjugateDir'*HessTimesConjugateDir;
    if(curvature <= 0)
        % compute scaled inexact trial step
        scaling = scaleFactor(stateReducedBasisCoeff,stateBasis,oldControl,newtonStep,conjugateDir,trustRegion);
        scaledNewtonStep = newtonStep + scaling * conjugateDir;
        break;
    end
    alpha = (oldDescentDir'*oldInvPrecTimesVector) / curvature;
    newtonStep = newtonStep + alpha * conjugateDir;
    if(itr == 1)
        cauchyStep = newtonStep;
    end
    normTrialStep = norm(newtonStep,2);
    if(normTrialStep >= trustRegion)
        % compute scaled inexact trial step
        scaling = scaleFactor(stateReducedBasisCoeff,stateBasis,oldControl,newtonStep,conjugateDir,trustRegion);
        scaledNewtonStep = newtonStep + scaling * conjugateDir;
        break;
    end
    newDescentDir = oldDescentDir + alpha*HessTimesConjugateDir;
    %newDescentDir = activeSet .* newDescentDir;
    normDescentDir = norm(newDescentDir,2);
    if(normDescentDir < tolerance)
        scaledNewtonStep = newtonStep;
        break;
    end
    newInvPrecTimesVector = ...
        applyInvPrecOperator(stateReducedBasisCoeff,stateBasis,oldControl,newDescentDir);
    rayleighQuotient = (newDescentDir'*newInvPrecTimesVector) ...
        / (oldDescentDir'*oldInvPrecTimesVector);
    conjugateDir = -newInvPrecTimesVector + rayleighQuotient*conjugateDir;
    %conjugateDir = activeSet .* conjugateDir;
    oldDescentDir = newDescentDir;
    oldInvPrecTimesVector = newInvPrecTimesVector;
    itr = itr + 1;
end

end

function [scaling] = ...
    scaleFactor(stateReducedBasisCoeff,stateBasis,control,newtonStep,conjugateDir,trustRegion)
[PrecTimesConjugateDir] = applyPrecOperator(stateReducedBasisCoeff,stateBasis,control,conjugateDir);
stepDotPrecTimesConjugateDir = newtonStep'*PrecTimesConjugateDir;
conjugateDirDotPrecTimesConjugateDir = ...
    conjugateDir'*PrecTimesConjugateDir;

[PrecTimesTrialStep] = applyPrecOperator(stateReducedBasisCoeff,stateBasis,control,newtonStep);
trialStepDotPrecTimesTrialStep = newtonStep'*PrecTimesTrialStep;

a = stepDotPrecTimesConjugateDir*stepDotPrecTimesConjugateDir;
b = conjugateDirDotPrecTimesConjugateDir ...
    *(trustRegion*trustRegion - trialStepDotPrecTimesTrialStep);
numerator = -stepDotPrecTimesConjugateDir + sqrt(a + b);
scaling = numerator / conjugateDirDotPrecTimesConjugateDir;
end

function [projCauchyStep, predRedBasedOnCauchyStep,activeSet] = ...
    projectedCauchyStep(objective,equality,oldState,stateReducedBasisCoeff,stateBasis,...
    oldControl,controlLowerBound, controlUpperBound,...
    oldGradient,trustRegion,maxProjItr,HFM)
alpha = 1;
mu0 = 1e-2;
mu1 = 1;
itr = 1;
contractionParam = 0.5;
while(1)
    alpha = -alpha;
    [projectedControl,activeSet] = ...
        project(oldControl,controlLowerBound,controlUpperBound,alpha,oldGradient);
    projCauchyStep = projectedControl - oldControl;
    oldGradDotProjCauchyStep = oldGradient'*projCauchyStep;
    sufficientDecreaseCondition = mu0 * oldGradDotProjCauchyStep;
    if(HFM == true)
        HessTimesProjCauchyStep = applyHessOperatorHFM(objective,equality,oldState,oldControl,projCauchyStep);
    else
        HessTimesProjCauchyStep = ...
            applyHessOperatorLFM(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,projCauchyStep);
    end
    predRedBasedOnCauchyStep = ...
        oldGradDotProjCauchyStep + 0.5*(projCauchyStep'*HessTimesProjCauchyStep);
    normProjCauchyStep = norm(projCauchyStep,2);
    if(predRedBasedOnCauchyStep <= sufficientDecreaseCondition && normProjCauchyStep <= mu1*trustRegion)
        break;
    elseif(itr >= maxProjItr)
        break;
    end
    itr = itr + 1;
    alpha = contractionParam*alpha;
end
end

function [newControl,predictedReduction,projTrialStep] = ...
    updateControl(objective,equality,oldState,stateReducedBasisCoeff,stateBasis,oldControl,...
    controlLowerBound,controlUpperBound,oldGradient,newTrialStep,...
    predRedBasedOnCauchyStep,trustRegion,maxItr,HFM)
alpha = 1;
mu0 = 1e-2;
mu1 = 1;
itr = 1;
while(1)
    % Compute projected control
    [newControl,~] = project(oldControl,controlLowerBound,controlUpperBound,alpha,newTrialStep);
    % Compute projected trial step
    projTrialStep = newControl - oldControl;
    % Compute reduced Hessian times trial step
    if(HFM == true)
        oldHessTimesProjTrialStep = applyHessOperatorHFM(objective,equality,oldState,oldControl,projTrialStep);
    else
        oldHessTimesProjTrialStep = ...
            applyHessOperatorLFM(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,projTrialStep);
    end
    % Evaluate quadratic model
    oldGradDotProjTrialStep = oldGradient' * projTrialStep;
    predictedReduction = ...
        (oldGradDotProjTrialStep + 0.5*(projTrialStep'*oldHessTimesProjTrialStep));
    % Evaluate sufficient decrease condition
    sufficientDecreaseCondition = mu0*predRedBasedOnCauchyStep;
    % Compute norm projected trial step
    normProjTrialStep = norm(projTrialStep,2);
    if(predictedReduction <= sufficientDecreaseCondition && normProjTrialStep <= mu1*trustRegion)
        break;
    elseif(itr > maxItr)
        break;
    end
    itr = itr+1;
    alpha = 0.5*alpha;
end
end

function [projControl,activeSet] = project(control, lowerBound, upperBound, alpha, direction)
projControl = control + alpha*direction;
nprimal = size(projControl,1);
activeSet = zeros(nprimal,1);
for i=1:nprimal
    activeSet(i) = ~((projControl(i) >= upperBound(i)) || (projControl(i) <= lowerBound(i)));
    projControl(i) = max(projControl(i),lowerBound(i));
   %activeSet(i) = (projControl(i) > lowerBound(i));
    projControl(i) = min(projControl(i),upperBound(i));
    %activeSet(i) = (projControl(i) < upperBound(i));
end
end

function [invPrec_times_vector] = applyInvPrecOperator(stateReducedBasisCoeff,stateBasis,control,vector)
invPrec_times_vector = vector;
end

function [Prec_times_vector] = applyPrecOperator(stateReducedBasisCoeff,stateBasis,control,vector)
Prec_times_vector = vector;
end

function [HessTimesVector] = applyHessOperatorLFM(objective,equality,stateReducedBasisCoeff,stateBasis,control,vector)
global GLB_INVP;
switch GLB_INVP.ProblemType
    case 'LP'
        [HessTimesVector] = ...
            lowFidelityHessTImesVecTypeLP(objective,equality,stateReducedBasisCoeff,stateBasis,control,vector);
    case 'NLP'
        [HessTimesVector] = ...
            highFidelityHessTImesVecTypeNLP(objective,equality,state,control,vector);
end
end

function [HessTimesVector] = applyHessOperatorHFM(objective,equality,state,control,vector)
global GLB_INVP;

switch GLB_INVP.ProblemType
    case 'LP'
        [HessTimesVector] = ...
            highFidelityHessTimesVecTypeLP(objective,equality,state,control,vector);
    case 'NLP'
        [HessTimesVector] = ...
            highFidelityHessTImesVecTypeNLP(objective,equality,state,control,vector);
end
end

function [HessTimesVector] =highFidelityHessTimesVecTypeLP(objective,equality,state,control,vector)
[objective_component] =  objective. secondDerivativeWrtControlControl(state,control,vector);
dual = zeros(size(state));
[equality_component] =  equality. secondDerivativeWrtControlControl(state,control,dual,vector);
HessTimesVector = objective_component + equality_component;
end


function [HessTimesVector] =lowFidelityHessTImesVecTypeLP(objective,equality,stateReducedBasisCoeff,stateBasis,control,vector)
[objective_component] =  objective. lowFidelitySecondDerivativeWrtControlControl(stateReducedBasisCoeff,stateBasis,control,vector);
dual = zeros(size(stateBasis,1),1);
[equality_component] =  equality. secondDerivativeWrtControlControlLF(stateReducedBasisCoeff,stateBasis,control,dual,vector);
HessTimesVector = objective_component + equality_component;
end