function [I] = evaluate_indic(r, w, h, Ny, h_flo )
I.w_s.reliability = 0;
I.w_s.vulnerability = 0;
I.w_s.deficit = 0;
I.w_s.resilience = 0;
I.flood.number = 0;
I.enviroment.low_pulses = 0;
I.enviroment.high_pulses = 0;

% reliability
I.w_s.reliability = sum( r >= w ) / length(r);

% vulnerability
def = max( w-r, 0 ) ;
I.w_s.vulnerability = sum(def) / sum( r < w );

% daily average squared deficit
I.w_s.deficit = mean( def.^2 );

% resilience
idxF = find( r<w ) ;
rec = 0;
for i=1:length(idxF)
    id_current = idxF(i) ;
    id_next = id_current+1 ;
    if ( id_current < length(r) ) && ( r(id_next) >= w )
        rec = rec+1 ;
    end
end
I.w_s.resilience = rec / sum( r < w );



% INDICATOR FOR FLOODING
% IF1 = mean annual number of days with flooding events
if h_flo ~= 0 
    I.flood.number = sum( h>h_flo )/Ny;
end 

% INDICATORS FOR ENVIRONMENT
global Gabcikovo;
LP = prctile( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_, 25 ); % thresholds are defined over the inflow
HP = prctile( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_, 75 ); % thresholds are defined over the inflow 

% IE1 = number of low pulses
I.enviroment.low_pulses = sum( r < LP ) / Ny;
% IE2 = number of high pulses
I.enviroment.high_pulses = sum( r > HP ) / Ny;

end