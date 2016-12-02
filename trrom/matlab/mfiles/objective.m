function [Operators] = objective()
Operators.value=@(state,control) value(state,control);
Operators.partialDerivativeState=@(state,control) partialDerivativeState(state,control);
Operators.partialDerivativeControl=@(state,control) partialDerivativeControl(state,control);
Operators.partialDerivativeControlState=@(state,control,vector) partialDerivativeControlState(state,control,vector);
Operators.partialDerivativeControlControl=@(state,control,vector) partialDerivativeControlControl(state,control,vector);
Operators.partialDerivativeStateState=@(state,control,vector) partialDerivativeStateState(state,control,vector);
Operators.partialDerivativeStateControl=@(state,control,vector) partialDerivativeStateControl(state,control,vector);
Operators.evaluateObjectiveInexactness=@(state,control) evaluateObjectiveInexactness(state,control);
Operators.evaluateGradientInexactness=@(state,control) evaluateGradientInexactness(state,control);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = value(state,control)
output = state'*control;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeState(state,control)
output = state.*control;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControl(state,control)
output = 2.*(state.*control);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlState(state,control,vector)
output = (state.*control) + vector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlControl(state,control,vector)
output = (state.*control) + 2.*vector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateState(state,control,vector)
output = (state.*control) + 3.*vector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateControl(state,control,vector)
output = (state.*control) + 4.*vector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluateObjectiveInexactness(state,control)
output = state'*control + 4;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluateGradientInexactness(state,control)
output = state'*control + 5;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%