% The input must be a function in the coefficient space
function plotSolution(usr_par,u)
    figure; clf
    trisurf(usr_par.mesh.t,usr_par.mesh.p(:,1), ...
	usr_par.mesh.p(:,2),u,'facecolor','interp')
    title(' u_{exp} (target / experimental / observed states) ');
    xlabel('x'); ylabel('y');
    %axis([-1 1 -1 1 -.1 .1]);
