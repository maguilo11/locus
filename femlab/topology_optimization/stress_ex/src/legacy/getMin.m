function [Output,Primal,TimeData] = getMin(Options, Operators)

global GLB_INVP;

Primal = [];
TimeData = [];
switch Options.SolutionType
    case 'NonLinearCG'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run NonLinearCG
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkNonLinearCG(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'QuasiNewton'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run the algorithm
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkQuasiNewton(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'NewtonTypeLS'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Newton-TypeLS
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkNewtonTypeLS(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'NewtonTypeTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Newton-TypeTR
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkNewtonTypeTR(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'IxNewtonTypeLS'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact Newton-TypeTR
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkIxNewtonTypeLS(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'IxNewtonTypeTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact Newton-TypeTR
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkIxNewtonTypeTR(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'LinMoreNewtonTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Lin-More Trust Region Algorithm
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkLinMoreTrustRegion(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'IxLinMoreNewtonTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact Lin-More Trust Region Algorithm
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkIxLinMoreTrustRegion(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'KelleySachsNewtonTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Kelley-Sachs Trust Region Algorithm
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkKelleySachsTrustRegion(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'IxKelleySachsNewtonTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact Kelley-Sachs Trust Region Algorithm 
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkIxKelleySachsTrustRegion(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'OptimalityCriteria'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact Newton-TypeTR
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkOptimalityCriteria(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'MMA'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact Newton-TypeTR
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkMMA(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Compute Final State
        [Primal.FinalState] = ...
            Operators.EqualityConstraint.solve(Output.Control);
    case 'IxSqpTypeTR'
        % Compute Initial State
        [Primal.InitialState] = ...
            Operators.EqualityConstraint.solve(Options.Control);
        % Run Inexact SQP-TypeTR
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        [Output] = mexDOTkIxSQPTypeTR(Options,Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        % Get Optimal Final State
        Primal.Solution = GLB_INVP.u_dirichlet;
        Primal.Solution(GLB_INVP.FreeNodes) = Output.State;
    case 'Diagnostics'
        TimeData.proctime = cputime;
        TimeData.walltime = tic;
        checkOperators(Options, Operators);
        TimeData.proctime = cputime - TimeData.proctime;
        TimeData.walltime = toc(TimeData.walltime);
        Output = [];
    otherwise
        error('\n **** Invalid Solution Type. See Users Manual. **** \n');
end

end