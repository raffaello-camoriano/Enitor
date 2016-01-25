function [ hMean ] = bandplot( data , color , alpha , log , numSD)
%BANDPLOT Plots mean and standard deviation of the values contained in the
%n x m 'data' matrix
%   INPUT
%   data: n x m matrix (n: number of repetitions; m: number of iterations)
%   color: color of the plot + fill (Standard MATLAB color name, e.g. 'blue' or 'red')
%   alpha: transparency parameter
%   log: 1 or 0
%   numSD: Number of standard deviations to be plotted
%
%   OUTPUT
%   hMean: Returns the handle to the mean value plot
%   hFill: Returns the handle to the fill plot

    m = mean(data)';
    sd = std(data,0)';
    f = [ m'+numSD*sd' , flipdim(m'-numSD*sd',2)]; 
    hold on;
    if log == 1
        hMean = semilogx(1:size(data,2) , m , 'Color' ,color , 'LineWidth',1);
        hFill = fill([1:size(data,2) , size(data,2):-1:1] , f, color, ...
            'FaceAlpha', alpha,'LineStyle','none');
        semilogx(1:size(data,2) , m , 'Color' , color , 'LineWidth',1);    
    else
        hMean = plot(1:size(data,2) , m ,'Color' ,  color , 'LineWidth',1);
        hFill = fill([1:size(data,2) , size(data,2):-1:1] , f, color, ...
            'FaceAlpha', alpha,'LineStyle','none');
        plot(1:size(data,2) , m , 'Color' , color , 'LineWidth',1);    
    end
end

