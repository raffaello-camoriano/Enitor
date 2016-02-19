function progressBar( currentIdx , total )
%PROGRESSBAR Draws a progress bar in the command window

    fprintf('[');

    for i = 1:20
        
        if floor(currentIdx *20 / total) > i
            fprintf('-');
        elseif floor(currentIdx *20 / total) == i
            fprintf('>');
        else
            fprintf(' ');
        end
    end
    
    fprintf(']');
    
end

