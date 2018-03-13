% Get all PDF files in the current folder
DIR = './training/female/';
files = dir(strcat(DIR,'*.jpg'));

% Loop through each
idx = 0;
for id = 1:length(files)
    % Get the file name (minus the extension)
    [~, f] = fileparts(files(id).name);
    movefile(strcat(DIR,files(id).name), strcat(DIR,sprintf('%d.jpg', idx)));
    idx = idx+1;
end