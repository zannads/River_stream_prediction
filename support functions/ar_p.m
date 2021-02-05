function [J_c_k, R2_c_k, J_v_k, R2_v_k] = ar_p(n_c, n_v, p)

%global Gabcikovo;
global T;

%calibration
        %compute m_c s_c
        %remove trend
        [x_c, m_c, s_c] = detrend(n_c, T.days, T.f);
        
        %evaluate param
        y = x_c( (p+1) : end );
        
        %M_c = p colonne, da 1 a n-p, 2 n-p-1..
        M_c = zeros( (length(x_c) - p ), p); 
        for j = 1 :p
            M_c(:, j) =  x_c( j : (end - p -1 + j) ) ;
        end
            
        theta = M_c \ y;
        
        %simu
        x_c__ = M_c * theta;
        x_c__ = [x_c(1:p); x_c__];
        
        %re-add the trend
        n_c__ = x_c__ .* s_c + m_c;
        
        %evaluate errors
        %MSE
        J_c_k = mean ( (n_c( p+1 : end ) - n_c__( p+1 : end ) ).^2 );
        
        %R2
        R2_c_k = 1 - sum( (n_c( p+1 : end ) - n_c__( p+1 : end )).^2 ) / sum( (n_c( p+1 : end ) - m_c(p+1:end)).^2 );
        
        %validation
        %remove trend
        [x_v, m_v, s_v] = detrend(n_v, T.days, T.f);
        
        %evaluate param
        %M_v = p colonne, da 1 a n-p, 2 n-p-1..
        M_v = zeros( (length(x_v) - p ), p); 
        for j = 1 :p
            M_v(:, j) =  x_v( j : (end - p -1 + j) ) ;
        end
        
        %simu, 
        x_v__ = M_v * theta;
        x_v__ = [x_v(1:p); x_v__];
        
        %re-add the trend
        n_v__ = x_v__ .* s_v + m_v;
        
        %evaluate errors
        %MSE
        J_v_k = mean ( (n_v( p+1 : end ) - n_v__( p+1 : end ) ).^2 );
        
        %R2
        R2_v_k = 1 - sum( (n_v( p+1 : end ) - n_v__( p+1 : end )).^2 ) / sum( (n_v( p+1 : end ) - m_v(p+1:end)).^2 );
end     