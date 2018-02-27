% Create either a histogram intersection kernel or a chi-squared kernel
% If useweights = 1, then different weights are given to the values coming
% from different levels in the spatial pyramid
function K = customkernel(X,Y,noperators,useweights,kerneltype)
weights = [ones(1,noperators)*(1/4),ones(1,noperators*4)*(1/2),ones(1,noperators*16)];
if useweights == 1
    if strcmp(kerneltype,'intersection')
        kernel = @(x,Y) pyramidmatch(x,Y,weights);
    elseif strcmp(kerneltype, 'chi2')
        kernel = @(x,Y) chi2(x,Y,weights);
    end
else
    if strcmp(kerneltype,'intersection')
        kernel = @(x,Y) sum(bsxfun(@min,x,Y),2);
    end       
end
K = pdist2(X,Y,kernel);

function s = pyramidmatch(x,Y,weights)
b = bsxfun(@min,x,Y);
t = bsxfun(@times,b,weights);
s = sum(t,2);

function kernel = chi2(x,Y,weights)
d = bsxfun(@minus, x, Y);
s = bsxfun(@plus, x, Y);
b = (d.^2) ./ (s/2 +eps);
t = bsxfun(@times,b,weights);
kernel = 1-sum(t, 2);