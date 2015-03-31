% A simple example for the utilisation of movie2gif function
clear all
close all
clc 

% we discretize [0, pi] interval
x = 0:0.1:2*pi;
y = sin(x);

figure('Position',[1 1 240 200])

for i = 1:length(x)
    plot(x(1:i),y(1:i), 'r', 'LineWidth',2)
    axis([0 2*pi -1.2 1.2])
    pause(0.01)
    mov(length(x)-i+1) = getframe;
end

movie2gif(mov, 'sin.gif', 'LoopCount', 0, 'DelayTime', 0)
