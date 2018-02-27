% Confifuring mingw follow this tutorial https://nl.mathworks.com/matlabcentral/fileexchange/52848-matlab-support-for-mingw-w64-c-c++-compiler
% setenv('MW_MINGW64_LOC','C:\TDM-GCC-64')
% mex -setup
% First compile the relevant c functions
function setup

% First, compile the c-function called maxblurring_written_in_c.c
d = dir('../COSFIRE/maxblurring.*');
if isempty(d)
    mex ../COSFIRE/written_in_c_maxblurring.c -outdir ../COSFIRE -output maxblurring;
end

% And compile the libsvm Matlab scripts
run('../libsvm_3_21/matlab/make.m');
