%CLEARALLBUTBP: Clears all the workspace but breakpoints

s=dbstatus;
save('myBreakpoints.mat', 's');
clear all
load('myBreakpoints.mat');
dbstop(s);
clear s;


