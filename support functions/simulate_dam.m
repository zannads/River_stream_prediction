function [s, h, r] = simulate_dam( n, h_init, param, mode ) 
delta = 60*60*24;
H = length(n)-1 ; 
% initialization of vectors
s = nan( size(n) );
h = nan( size(n) );
r = nan( size(n) );
% initial condition t=1
h(1) = h_init; 
s(1) = h_init * param.nat.S ;

if( strcmp(mode ,'reg') )
    for t=1:H
        % 1)release
        r(t+1) = regulated_release( param , h(t) );
         % 2) mass-balance
        s(t+1) = s(t) + ( n(t+1) - r(t+1) )*delta ;
        % 3) s->h
        h(t+1) = s(t+1)/param.nat.S ;
    end
else 
    for t=1:H
        % 1)release
        r(t+1) = n(t+1);  
        % 2) mass-balance
        s(t+1) =  n(t+1)*delta ;
        % 3) s->h
        h(t+1) = s(t+1)/param.nat.S ;
    end
end
 