function params = Parameters(type)
% VERSION 28/05/2012
% CREATED BY: George Azzopardi and Nicolai Petkov, University of Groningen,
%             Johann Bernoulli Institute for Mathematics and Computer Science, Intelligent Systems
%
% SystemConfig returns a structure of the parameters required by the
% COSFIRE operator

switch (type)
    case 1
        params = ParametersSS;
    case 2
        params = ParametersTS;
    otherwise
        % define your own parameters that is appropriate for your
        % application
end