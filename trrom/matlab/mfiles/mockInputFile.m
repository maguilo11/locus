function [data,default] = mockInputFile
% Integer data
data.NumberDuals = 10;
data.NumberSlacks = 11;
data.NumberStates = 20;
data.NumberControls = 15;
data.MaxNumberSubProblemIterations = 5;
data.MaxNumberOuterIterations = 30;
% Scalar data
data.MinTrustRegionRadius = 1e-1;
data.MaxTrustRegionRadius = 1e2;
data.TrustRegionContractionScalar = 0.22;
data.TrustRegionExpansionScalar = 4;
data.ActualOverPredictedReductionMidBound = 0.3;
data.ActualOverPredictedReductionLowerBound = 0.15;
data.ActualOverPredictedReductionUpperBound = 0.8;
data.StepTolerance = 0.14;
data.GradientTolerance = 0.13;
data.ObjectiveTolerance = 0.12;
data.StagnationTolerance = 0.11;
% Array data
data.ControlLowerBound = 0.1 .* ones(10,1);
data.ControlUpperBound = ones(10,1);
%% DEFAULT struct
default.Default = 0;
end