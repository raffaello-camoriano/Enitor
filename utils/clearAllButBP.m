%CLEARALLBUTBP: Script which clears all the workspace but breakpoints

s=dbstatus;
save('myBreakpoints.mat', 's');
clear all
load('myBreakpoints.mat');
delete('myBreakpoints.mat');
dbstop(s);
clear s;


