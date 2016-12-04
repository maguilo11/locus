function [solution] = ...
    prototype(gradTol,stepTol,ObjFuncTol,maxOptItr,maxSubProbItr,basisTreshold)
% Prototype for reduced model based topology optimization
clc;

% Problem interface directories
addpath ./interface2/;
addpath /scratch/maguilo/research/matlab/intrelab/tools/;
addpath /scratch/maguilo/research/mathvault/parest/code/intrelab;
addpath /scratch/maguilo/research/mathvault/parest/code/intrelab/mesh2;

global GLB_INVP;

fprintf('\n*** Elastosatics Topology Optimization ***\n');

% Domain specifications
Domain.xmin = 0;  % min dim in x-dir
Domain.xmax = 1.5;   % max dim in x-dir
Domain.ymin = 0;  % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 30;     % num intervals in x-dir
Domain.ny = 20;     % num intervals in y-dir
% Regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
beta = 0.5*(Domain.xmax / (Domain.nx)); 
%beta = 1e-5;
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
GLB_INVP.alpha = 1 / (sum(GLB_INVP.Ms*InitialControl)^2);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
GLB_INVP.MinCellStifsnessMat = computeStiffnessMatrix(GLB_INVP.Gmin, GLB_INVP.Bmin);
[state] = equality.solve(one);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = 0.5 / (state'*K*state);
clear K;

% Solve reduced order model based optimization problem
[solution,fval,grad,itr,trialStep] = ...
    getMin(objective,equality,InitialControl,gradTol,stepTol,ObjFuncTol,maxOptItr,maxSubProbItr,basisTreshold);

% Plot Results
solution = -1*solution;
plotResults(solution);
end

function [control,newFval,newGrad,projTrialStep,itr] = ...
    getMin(objective,equality,control,gradTol,stepTol,ObjFuncTol,maxItr,maxSubProbItr,basisTreshold)
% Solve pde constraint
[state] = equality.solve(control);
% Compute objective function
oldFval = objective.evaluate(state,control);
% Build initial orthogonal basis
[stateBasis] = buildOrthogonalBasis(state,basisTreshold);
% Solve reduced pde constraint
[stateReducedBasisCoeff] = equality.lowFidelitySolve(stateBasis,control);
% Assemble reduced gradient
[oldGrad] = objective.lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,control);
% Solve optimization problem
itr = 0;
oldTrustRegion = norm(oldGrad,2);
while(1)
    [control,newFval,snapShots,newTrustRegion,projCauchyStep,projTrialStep,updateBasis] = ...
        trustRegionSubProblem(objective,equality,stateReducedBasisCoeff,stateBasis,...
        control,oldFval,oldGrad,oldTrustRegion,maxSubProbItr);
    if(updateBasis == true)
        % update orthogonal basis
        [stateBasis] = buildOrthogonalBasis(snapShots,basisTreshold);
        % Solve reduced pde constraint
        [stateReducedBasisCoeff] = equality.lowFidelitySolve(stateBasis,control);
    end
    % Compute new gradient
    [newGrad] = objective.lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,control);
    % Check convergence
    deltaObjectiveFunction = abs(oldFval-newFval);
    infNormCauchyStep = norm(projCauchyStep,Inf);
    normProjTrailStep = norm(projTrialStep,2);
    if(infNormCauchyStep <= gradTol)
        break;
    elseif(normProjTrailStep <= stepTol)
        break;
    elseif(deltaObjectiveFunction <= ObjFuncTol)
        break;
    elseif(itr >= maxItr)
        break;
    end
    itr = itr + 1;
    plotResults(control);
    oldFval = newFval;
    oldGrad = newGrad;
    oldTrustRegion = newTrustRegion;
end

end

function [basis] = buildOrthogonalBasis(snapShots,threshold)
% Compute eigenvectors and eigenvalues
A = snapShots'*snapShots;
B = speye(size(A));
[eigenvectors,D] = eigs(A,B,length(A));
eigenvalues = diag(D);
% Gather eigenvector set to store based on the total energy
stop = false;
totalEigenValueSum = sum(eigenvalues);
numberBasisFuncs = 0;
while(stop == false)
    numberBasisFuncs = numberBasisFuncs + 1;
    energy = sum(eigenvalues(1:numberBasisFuncs))/totalEigenValueSum;
    if(energy >= threshold)
        stop = true;
    end
end
% Compute orthogonal basis
basis=snapShots*eigenvectors(:,1:numberBasisFuncs);
for i=1:numberBasisFuncs
    const = 1 / sqrt(eigenvalues(i));
    basis(:,i) = const .* basis(:,i);
end
end

function [newControl,newFval,snapShots,trustRegion,projCauchyStep,projTrialStep,updateBasis] = ...
    trustRegionSubProblem(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,...
                                                           oldFval,oldGradient,trustRegion,maxItr)
itr = 1;
maxPcgItr = 200;
maxProjectionItr = 10;
updateBasis = true;
controlUpperBound = ones(size(oldControl));
controlLowerBound = 1e-2*ones(size(oldControl));
tolerance = 1e-1*norm(oldGradient);
trustRegionReduction = 0.5;
trustRegionExpansion = 2.;
trustRegionLowerBound = 0.2;
trustRegionMiddleBound = 0.4;
trustRegionUpperBound = 0.8;
numState = size(stateBasis,1);
newsnapShots = zeros(numState,maxItr);
while(1)
    % Compute projected Cauchy step
    [projCauchyStep,predRedBasedOnCauchyStep,activeSet] = ...
        projectedCauchyStep(objective,equality,stateReducedBasisCoeff,stateBasis, ...
                                                          oldControl,controlLowerBound,controlUpperBound,...
                                                          oldGradient,trustRegion,maxProjectionItr);
    % Compute newton step
    [scaledNewtonStep] = ...
        steihaugTointCg(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,...
                                              oldGradient,trustRegion,tolerance,maxPcgItr,activeSet);
    % Update control
    [newControl,predictedReduction,projTrialStep] = ...
        updateControl(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,...
                                          controlLowerBound,controlUpperBound,oldGradient,scaledNewtonStep,...
                                          predRedBasedOnCauchyStep,trustRegion,maxProjectionItr);
    % Solve pde constraint
    [newState] = equality.solve(newControl);
    % Compute objective function
    [newFval] = objective.evaluate(newState,newControl);
    % Collect snapshot
    newsnapShots(:,itr) = newState;
    % Compute actual reduction based on new control and state values 
    actualReduction = newFval - oldFval;
    % Compute actual over predicted reduction ratio
    rho = actualReduction / predictedReduction;
    if(rho >= trustRegionUpperBound)
        updateBasis = false;
        snapShots =  newsnapShots(:,1:itr);
        trustRegion = trustRegionExpansion * trustRegion;
        break;
    elseif(rho >= trustRegionMiddleBound && rho < trustRegionUpperBound)
        updateBasis = true;
        snapShots =  newsnapShots(:,1:itr);
        break;
    elseif(rho >= trustRegionLowerBound && rho < trustRegionLowerBound)
        updateBasis = true;
        snapShots =  newsnapShots(:,1:itr);
        break;
    elseif(itr > maxItr)
        updateBasis = true;
        snapShots =  newsnapShots(:,1:itr);
        break;
    else
        trustRegion = trustRegionReduction * trustRegion;
    end
    itr = itr+1;
end

end

function [scaledNewtonStep, newtonStep, cauchyStep] = ...
    steihaugTointCg(objective,equality,stateReducedBasisCoeff,stateBasis,...
                                          control,gradient,trustRegion,tolerance,maxItr,activeSet)
% initialize newton step
newtonStep = zeros(size(control));
% initialize descent direction
oldDescentDir = activeSet .* gradient;
% Apply preconditioner
oldInvPrecTimesVector = ...
    applyInvPrecOperator(stateReducedBasisCoeff,stateBasis,control,oldDescentDir);
% Conjugate direction
conjugateDir = -oldInvPrecTimesVector;
% Start Krylov solver
itr = 1;
while(1)
    if(itr >= maxItr)
        scaledNewtonStep = newtonStep;
        break;
    end
    % Apply direction to Hessian operator
    conjugateDir = activeSet .* conjugateDir;
    HessTimesConjugateDir = ...
        applyHessOperator(objective,equality,stateReducedBasisCoeff,stateBasis,control,conjugateDir);
    curvature = conjugateDir'*HessTimesConjugateDir;
    if(curvature <= 0)
        % compute scaled inexact trial step
        scaling = scaleFactor(stateReducedBasisCoeff,stateBasis,control,newtonStep,conjugateDir,trustRegion);
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
        scaling = scaleFactor(stateReducedBasisCoeff,stateBasis,control,newtonStep,conjugateDir,trustRegion);
        scaledNewtonStep = newtonStep + scaling * conjugateDir;
        break;
    end
    newDescentDir = oldDescentDir + alpha*HessTimesConjugateDir;
    normDescentDir = norm(newDescentDir,2);
    if(normDescentDir < tolerance)
        scaledNewtonStep = newtonStep;
        break; 
    end
    newDescentDir = activeSet .* newDescentDir;
    newInvPrecTimesVector = ...
        applyInvPrecOperator(stateReducedBasisCoeff,stateBasis,control,newDescentDir);
    rayleighQuotient = (newDescentDir'*newInvPrecTimesVector) ...
        / (oldDescentDir'*oldInvPrecTimesVector);
    conjugateDir = -newInvPrecTimesVector + rayleighQuotient*conjugateDir;
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
    projectedCauchyStep(objective,equality,reducedBasisCoeff,stateBasis,oldControl,controlLowerBound,...
                                                     controlUpperBound,oldGradient,trustRegion,maxProjItr)
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
    HessTimesProjCauchyStep = ...
        applyHessOperator(objective,equality,reducedBasisCoeff,stateBasis,oldControl,projCauchyStep);
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
    updateControl(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,...
                                      controlLowerBound,controlUpperBound,oldGradient,newTrialStep,...
                                      predRedBasedOnCauchyStep,trustRegion,maxItr)
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
    [oldHessTimesProjTrialStep] = ...
        applyHessOperator(objective,equality,stateReducedBasisCoeff,stateBasis,oldControl,projTrialStep);
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
activeSet = ones(nprimal,1);
for i=1:nprimal
    projControl(i) = max(projControl(i),lowerBound(i));
    activeSet(i) = (projControl(i) > lowerBound(i));
    projControl(i) = min(projControl(i),upperBound(i));
    activeSet(i) = (projControl(i) < upperBound(i));
end
end

function [invPrec_times_vector] = applyInvPrecOperator(stateReducedBasisCoeff,stateBasis,control,vector)
invPrec_times_vector = vector;
end

function [Prec_times_vector] = applyPrecOperator(stateReducedBasisCoeff,stateBasis,control,vector)
Prec_times_vector = vector;
end

function [HessTimesVector] = applyHessOperator(objective,equality,stateReducedBasisCoeff,stateBasis,control,vector)
%[HessTimesVector] =  objective. lowFidelitySecondDerivativeWrtControlControl(stateReducedBasisCoeff,stateBasis,control,vector);
[objective_component] =  objective. lowFidelitySecondDerivativeWrtControlControl(stateReducedBasisCoeff,stateBasis,control,vector);
dual = zeros(size(stateBasis,1),1);
[equality_component] =  equality. secondDerivativeWrtControlControlLF(stateReducedBasisCoeff,stateBasis,control,dual,vector);
HessTimesVector = objective_component + equality_component;
end
