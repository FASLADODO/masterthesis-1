% Apply L2-normalization to each tile. Each tile contains as many values as
% the number of operators
function data = normalizeData(data,noperators)
fun = @(x) normr(x);
data.training.desc = blkproc(data.training.desc,[size(data.training.desc,1),noperators],fun);
data.testing.desc = blkproc(data.testing.desc,[size(data.testing.desc,1),noperators],fun);