function r = regulated_release( param , h )



% LAKE MODELS PARAMETERS

% natural storage-discharge relationship linear
m     = param.nat.m   ; 
h0       = param.nat.h0     ; % [m]

% regulated storage-discharge relationship
h_min    = param.reg.h_min  ; % [m]
h1       = param.reg.h1     ; % [m]
h2       = param.reg.h2     ; % [m]
h_max    = param.reg.h_max  ; % [m]
m1       = param.reg.m1     ; % [m2/s]
m2       = param.reg.m2     ; % [m2/s]
w        = param.reg.w      ; % [m3/s]


% COMPUTATION OF THE LAKE RELEASE

% 1) regulated storage-discharge relationship
% water saving
L1 = w + m1 * ( h - h1 ) ; 
% floods control
L2 = w + m2 * ( h - h2 ) ;
% release
r  = max( [ min( L1 , w ) ; L2 ] )  ;

% 2) normative constrains
r( h <= h_min ) = 0                                   ;    % completely closed dam gates
r( h >= h_max ) = ( h( h >= h_max ) - h0 )*m ;    % completely open dam gates

% 3) physical constrains --> the values of the parameters h1, h2, m1, m2, w might generate
% release values that are not admissible from a physical point of view

% the release can not be negative
r( r < 0 )  = 0                            ; 
% find the release values larger than the maximum admissible outflow 
idx         = r > ( h - h0 )*m ;
r( idx )    = ( h( idx ) - h0 )*m ; 
% the release must be 0 if h(t) < h0 (this constrain is necessary if
% 'h_min' is lower than 'h0')
r( h < h0 ) = 0  ; 
end 
                                             
