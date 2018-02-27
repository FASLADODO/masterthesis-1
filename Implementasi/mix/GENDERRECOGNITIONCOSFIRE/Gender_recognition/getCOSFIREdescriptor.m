% Create a COSFIRE descriptor using a spatial pyramid of two levels.
%    Level 0 - we consider the whole COSFIRE response map as one tile
%    Level 1 - we use 2 x 2 tiling
%    Level 2 - we use 4 x 4 tiling
% In each tile we take the maximum COSFIRE response. The descriptor results
% in a 21-element feature vector
function desc = getCOSFIREdescriptor(imageset,operatorlist)
noperators = numel(operatorlist);
desc = zeros(size(imageset,3),noperators*21);

% pointer to a function that takes the maximum in a given tile
fun = @(x) max(x(:));

for i = 1:size(imageset,3)
    fprintf('Compute descriptor %d of %d\n',i,size(imageset,3)); 

    % Compute the Gabor response maps that are defined in the tuples of all
    % COSFIRE filter. These responses are stored in a hashtable and are
    % shared among all COSFIRE filters.
    tuple = computeTuples(imageset(:,:,i)./255,operatorlist);
    
    for j = 1:numel(operatorlist)
        % First compute the COSFIRE response map
        output = applyCOSFIRE(imageset(:,:,i),operatorlist{j},tuple);
        
        % For each of the three levels in the spatial pyramid compute the
        % maximum COSFIRE response in each tile.
        for k = 0:2
            tilewidth = 128/(2^k);
            maxout{k+1} = blkproc(output,[tilewidth tilewidth],fun);
        end
        idx = j:noperators:noperators*21;
        
        % Organize the 21 values in one vector
        v = vertcat(maxout{1}(:),maxout{2}(:),maxout{3}(:))';
        
        % Place the values of this vector in the corresponding bins of the
        % desriptor matrix of all images.
        desc(i,idx) = v;
    end
end