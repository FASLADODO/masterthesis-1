% Configures a list of COSFIRE filters by randomly selected regions from
% randomly selected training images.
%
% Input parameters:
%   imageset: the training data set
%   noperators: Total number of COSFIRE filters to be configured
%   noperatorsperface: Number of COSFIRE filters to be configured by
%   randomly selected regions of the same face

function cosfirelist = configureCOSFIREfilters(imageset,noperators,noperatorsperface)

params = ParametersGender;

permlist = randperm(size(imageset,3),noperators);

for i = 1:numel(permlist)
    fprintf('configuring cosfire %d of %d\n',i,numel(permlist));
    im = imageset(:,:,permlist(i));    
    for j = 1:noperatorsperface
        cosfire{j} = [];
        
        % If the number of tuples is less than 5 then the COSFIRE filter is
        % ignored. This is because such a filter will have low selectivity
        while isempty(cosfire{j}) || size(cosfire{j}.tuples,2) < 5
            try
                location = ceil(rand(1,2) .* [size(imageset,1),size(imageset,2)]);
                cosfire{j} = configureCOSFIRE(im./255,location,params);
            catch ME
                % location is out of bounds and threfore it is ignored
            end
            rng('shuffle');
        end    
    end
    cosfirelist(noperatorsperface*(i-1)+1:noperatorsperface*i) = cosfire;
end

% Set the threshold t1 of the input Gabor responses to 0
for i = 1:numel(cosfirelist)
    cosfirelist{i}.params.COSFIRE.t1 = 0;
end
