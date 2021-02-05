function [x, m, s] = detrend(n, days, f)
    
    [ mi, m ] = moving_average( n , days , f ) ;
    [ sigma2 , s2 ] = moving_average( ( n - m ).^2 , days , f ) ;
    %sigma = sigma2 .^ 0.5;
    s = s2 .^ 0.5;
    x = (n - m) ./ s;

end