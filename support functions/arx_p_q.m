function [R2_c_pro, R2_v_pro] = arx_p_q(n_c, i_c, n_v, i_v, p, q, zpro_imp)
    %global Gabcikovo;
    global T;
    
    %remove trend to n
    [x_c, m_c, s_c] = detrend(n_c, T.days, T.f);
    [x_v, m_v, s_v] = detrend(n_v, T.days, T.f);
      
    %remove trend to every column of u
    n_input = size(i_c);
    n_input = n_input(2);
    u_c = zeros(size(i_c));
    u_v = zeros(size(i_v));
    for idx = 1:n_input
        %take idx column from i
        %detrend it 
        %add it to idx column of u 
        [uidx_c, ~, ~] =detrend(i_c( :, idx) , T.days, T.f);
        [uidx_v, ~, ~] =detrend(i_v( :, idx) , T.days, T.f);
        
        u_c (:, idx) = uidx_c;
        u_v (:, idx) = uidx_v;
    end
    
    %proper or improper?
    z = strcmp('improper' , zpro_imp);
    
    %building Y and M
    r = max(p, q);
    y_c = x_c( (r+1) : end );
    
    %M_c = N_c - (max p o q) rows, p coloumns for x q for each input
    M_c_p = zeros( (length(x_c) - r ), p );
    M_c_q = zeros( (length(x_c) - r ), n_input*q);
    for j = 1 : p
        M_c_p(:, j) =  x_c( r+1 - j : (end - j) ) ;
    end
    
    for i = 1: n_input
        for j = 1 : q
            M_c_q(:, j + q*(i-1) ) = u_c( (r+1 - j + z) : (end - j + z) , i );
        end
    end
    
    M_c = [M_c_p, M_c_q];
    
    %calculation param thet = M\Y
    theta_pro = M_c\y_c;
    
    %simu calib
    x_c__pro = [x_c(1:r); M_c * theta_pro];
    n_c__pro = x_c__pro.*s_c + m_c;
    
    %building M_v
    %M_v = N_v - (max p o q) rows, p coloumns for x q for each input
    M_v_p = zeros( (length(x_v) - r ), p );
    M_v_q = zeros( (length(x_v) - r ), n_input*q);
    for j = 1 : p
        M_v_p(:, j) =  x_v( r+1 - j : (end - j) ) ;
    end
    
    for i = 1: n_input
        for j = 1 : q
            M_v_q(:, j + q*(i-1) ) = u_v( (r+1 - j + z) : (end - j + z) , i );
        end
    end
    
    M_v = [M_v_p, M_v_q];
    
    %simu M_v
    x_v__pro = [x_v(1:r); M_v * theta_pro];
    n_v__pro = x_v__pro.*s_v + m_v;
    
    R2_c_pro    = 1 - sum( (n_c( r+1 : end ) - n_c__pro( r+1 : end )).^2 ) / sum( (n_c( r+1 : end )-m_c( r+1 :end) ).^2 );
    R2_v_pro    = 1 - sum( (n_v( r+1 : end ) - n_v__pro( r+1 : end )).^2 ) / sum( (n_v( r+1 : end )-m_v( r+1 :end) ).^2 );
    
    
    
end


%for 1 input or multiple input but q = 1
%     [u_c, mu_c, su_c] =detrend(i_c, T.days, T.f);
%     [u_v, mu_v, su_v] =detrend(i_v, T.days, T.f);
%     

%     % proper model:
%     y_c = x_c(2:end) ;
%     M_c = [ x_c(1:end-1) u_c(1:end-1 , : ) ] ;  
%     theta_pro    = M_c \ y_c;
%     
%     x_c__pro = [ x_c(1); M_c * theta_pro ] ;
%     n_c__pro = x_c__pro .* s_c + m_c ;
%    
%     M_v = [ x_v(1:end-1) u_v(1:end-1 , : ) ];
%     
%     x_v__pro = [ x_v(1); M_v * theta_pro ];
%     n_v__pro = x_v__pro .* s_v + m_v;
%     
%     R2_c_pro    = 1 - sum( (n_c( 2 : end ) - n_c__pro( 2 : end )).^2 ) / sum( (n_c( 2 : end )-m_c(2:end) ).^2 );
%     R2_v_pro    = 1 - sum( (n_v( 2 : end ) - n_v__pro( 2 : end )).^2 ) / sum( (n_v( 2 : end )-m_v(2:end) ).^2 );
%     
% % improper model:
% M_c = [ x(1:end-1) u(2:end) ] ; 
% y_c = x(2:end) ;
% theta_imp    = M_c \ y_c 
% x_arx_imp = [ x(1); M_c * theta_imp ] ;
% q_arx_imp = x_arx_imp .* s_q + m_q ;
% R2_imp    = 1 - sum( (q( 2 : end ) - q_arx_imp( 2 : end )).^2 ) / sum( (q( 2 : end )-m_q(2:end) ).^2 )
