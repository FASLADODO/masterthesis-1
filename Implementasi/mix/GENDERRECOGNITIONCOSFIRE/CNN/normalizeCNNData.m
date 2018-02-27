% Apply L2-normalization to each tile. Each tile contains as many values as
% the number of operators
function data = normalizeCNNData(data)
fun = @(x) normr(x);
noperators = length(data.training.features(1,:));
data.training.features = blkproc(data.training.features,[size(data.training.features,1),noperators],fun);
data.testing.features = blkproc(data.testing.features,[size(data.testing.features,1),noperators],fun);