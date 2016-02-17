function [ hMean ] = bandplot( X , Y , color , alpha , log , numSD , lineStyle)
%BANDPLOT Plots mean and standard deviation of the values contained in the
%n x m 'data' matrix
%   INPUT
%   X: X axis data (independent variable)
%   Y: n x m matrix (n: number of repetitions; m: number of iterations)
%   color: color of the plot + fill (Standard MATLAB color name, e.g. 'blue' or 'red')
%   alpha: transparency parameter
%   log: 1 or 0
%   numSD: Number of standard deviations to be plotted
%   lineStyle: standard linespec properties (e.g. '--')
%
%   OUTPUT
%   hMean: Returns the handle to the mean value plot
%   hFill: Returns the handle to the fill plot

    if isempty(X)
        X = 1:size(Y,2);
    end
    if isempty(lineStyle)
        lineStyle = '-';
    end

    m = mean(Y)';
    sd = std(Y,0)';
    f = [ m'+numSD*sd' , flipdim(m'-numSD*sd',2)]; 
    hold on;
    if log == 1
        hMean = semilogx(X , m , lineStyle, 'Color' ,color , 'LineWidth',1 );
        hFill = fill([X , flip(X)] , f, color, ...
            'FaceAlpha', alpha,'LineStyle','none');
        semilogx(X , m , lineStyle, 'Color' , color , 'LineWidth',1);    
        set(gca, 'xscale', 'log')
    else
        hMean = plot(X , m , lineStyle,'Color' ,  color , 'LineWidth',1);
        hFill = fill([X , flip(X)] , f, color, ...
            'FaceAlpha', alpha,'LineStyle','none');
        plot(X , m , lineStyle, 'Color' , color , 'LineWidth',1);    
    end
    hold off
end

