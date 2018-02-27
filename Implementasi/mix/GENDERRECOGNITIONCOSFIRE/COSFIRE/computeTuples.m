function tuple = computeTuples(inputImage,operator)

if ~iscell(operator)
    operatorlist{1} = operator;
else
    operatorlist = operator;
end

set = cell(0);
index = 1;

% First, populate a set of all the tuples produced by reflecting, rotating
% and scaling the operator
for op = 1:length(operatorlist)
    params = operatorlist{op}.params;
    reflectOperator = operatorlist{op};
                 
    for reflection = 1:2^params.invariance.reflection    
        if reflection == 2            
            if params.inputfilter.symmetric == 1                
                reflectOperator.tuples(2,:) = mod(pi - reflectOperator.tuples(2,:),pi);                
            else
                reflectOperator.tuples(2,:) = mod(pi - reflectOperator.tuples(2,:),2*pi);
            end            
            reflectOperator.tuples(4,:) = mod(pi - reflectOperator.tuples(4,:),2*pi);
        end        

        rotateOperator = reflectOperator;
        for psiindex = 1:length(params.invariance.rotation.psilist)
            if strcmp(params.inputfilter.name,'Gabor')
                if params.inputfilter.symmetric == 1
                    rotateOperator.tuples(2,:) = mod(reflectOperator.tuples(2,:) + params.invariance.rotation.psilist(psiindex),pi);
                else
                    rotateOperator.tuples(2,:) = mod(reflectOperator.tuples(2,:) + params.invariance.rotation.psilist(psiindex),2*pi);
                end
            end
            rotateOperator.tuples(4,:) = mod(reflectOperator.tuples(4,:) + params.invariance.rotation.psilist(psiindex),2*pi);            

            scaleOperator = rotateOperator;
            for upsilonindex = 1:length(params.invariance.scale.upsilonlist)
                % Scale the values of parameters lambda and rho of every tuple by a given upsilon value                                
                scaleOperator.tuples(1,:) = rotateOperator.tuples(1,:) * params.invariance.scale.upsilonlist(upsilonindex);                   
                scaleOperator.tuples(3,:) = rotateOperator.tuples(3,:) * params.invariance.scale.upsilonlist(upsilonindex);

                if strcmp(params.COSFIRE.outputfunction,'weightedgeometricmean')
                    tupleweightsigma = sqrt(-max(scaleOperator.tuples(3,:))^2/(2*log(params.COSFIRE.mintupleweight)));
                    scaleOperator.tuples(5,:) = exp(-scaleOperator.tuples(3,:).^2./(2*tupleweightsigma*tupleweightsigma));
                else
                    scaleOperator.tuples(5,:) = ones(size(scaleOperator.tuples(3,:))); 
                end
                set{index} = scaleOperator.tuples;
                index = index + 1;
            end    
        end
    end
end

S = cell2mat(set);
S = round(S * 10000) / 10000;
inputfilterparams = unique(S(1:2,:)','rows'); % A unique set of (lambda, theta)
tupleparams = unique(S([1:3,5],:)','rows'); % A unique set of (lambda, theta, rho)
sz = size(inputImage);

inputfilterresponse = zeros(size(inputImage,1),size(inputImage,2),size(inputfilterparams,1));
ninputfilterparams  = size(inputfilterparams,1);
hashkeylist         = cell(1,ninputfilterparams);
hashvaluelist       = cell(1,ninputfilterparams);

% Compute the responses of the bank of input filters
for i = 1:ninputfilterparams
    hashkeylist{i} = getHashkey(inputfilterparams(i,:));
    hashvaluelist{i} = i;                        
    inputfilterresponse(:,:,i) = getGaborResponse(inputImage,params.inputfilter.Gabor,inputfilterparams(i,1),inputfilterparams(i,2));    
end
inputfilterhashtable = containers.Map(hashkeylist,hashvaluelist);

% Threshold the responses by a fraction t1 of the maximum response
inputfilterresponse(inputfilterresponse < params.COSFIRE.t1*max(inputfilterresponse(:))) = 0;

ntupleparams = size(tupleparams,1);
hashkeylist = cell(1,ntupleparams);
hashvaluelist = cell(1,ntupleparams);

% Compute the blurred responses of the input filters
for i = 1:ntupleparams
    rho    = tupleparams(i,3);

    index = inputfilterhashtable(getHashkey(tupleparams(i,1:2)));
    hashkeylist{i} = getHashkey(tupleparams(i,1:3));    

    ifresp = inputfilterresponse(:,:,index);

    if any(ifresp(:))               
        % Compute the sigma of the 2D Gaussian function that will be
        % used to blur the corresponding output.
        sigma = (params.COSFIRE.sigma0 + (params.COSFIRE.alpha*rho));                

        if strcmp(params.COSFIRE.blurringfunction,'max')                    
            hashvaluelist{i} = maxblur(ifresp,sigma);
        elseif strcmp(params.COSFIRE.blurringfunction,'sum')
            blurfunction = fspecial('gaussian',[1 round(sigma.*6)],sigma);
            hashvaluelist{i} = conv2(blurfunction,blurfunction,ifresp,'same');
        end
    else
        hashvaluelist{i} = zeros(sz);
    end

    if strcmp(params.COSFIRE.outputfunction,'weightedgeometricmean')                                     
        % weight the blurred responses by a factor that depends on the
        % distance from the support center of the COSFIRE filter
        hashvaluelist{i} = hashvaluelist{i} .^ tupleparams(i,4);                    
    end    
end
tuple.hashtable  = containers.Map(hashkeylist,hashvaluelist); 