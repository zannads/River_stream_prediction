function [R2_opt, net_opt ] = ann_p( n, i, p)

global T;

[x, mx, sx] = detrend(n , T.days, T.f);
%remove trend to every column of u
n_input = size(i);
n_input = n_input(2);
u = zeros(size(i));
for idx = 1:n_input
    %take idx column from i
    %detrend it
    %add it to idx column of u
    [uidx_c, ~, ~] =detrend(i( :, idx) , T.days, T.f);

    u (:, idx) = uidx_c;
end

%[u, ~, ~] =detrend(i , T.days, T.f);

% ANN - proper model
X_pro = [ x(1:end-1) u(1:end-1, :) ]' ; 

Y = x(2:end)' ;

N_runs = 10 ;
R2_i_pro = zeros(N_runs,1);
for i = 1:N_runs
    net_i = newff(X_pro,Y, p) ; % initialization of ANN
    net_i = train( net_i, X_pro, Y ) ;  %training of ANN
    
    Y__pro =  net_i ( X_pro ) ;
    
    Y__pro = [ x(1); Y__pro' ] ;
    n__pro = Y__pro .* sx + mx ;
    
    R2_i_pro(i) = 1 - sum( (n( 2 : end ) - n__pro( 2 : end )).^2 ) / sum( (n( 2 : end )-mx(2:end) ).^2 );
    if R2_i_pro(i) >= max(R2_i_pro)
        net_opt = net_i ;       %do i need to return it??
        R2_opt = R2_i_pro(i);
    end
    
end

% 
% 
% net = newff(X,Y_c,3) ; % initialization of ANN
% net = train( net, X, Y_c ) ;
% Y_ = sim( net, X ) ;
% Y_pro = [ x(1); Y_' ] ;
% q_ann_pro = Y_pro .* s_q + m_q ;
% 
% R2_ann_pro = 1 - sum( (q( 2 : end ) - q_ann_pro( 2 : end )).^2 ) / sum( (q( 2 : end )-m_q(2:end) ).^2 )
% 
% % ANN - improper model
% X_imp = [ x_c(1:end-1) u_c(2:end) ]' ;
% 
% net_imp = newff(X,Y_c,3) ; % initialization of ANN
% net_imp = train( net_imp, X, Y_c ) ;
% Y_ = sim( net_imp, X ) ;
% Y_imp = [ x(1); Y_' ] ;
% q_ann_imp = Y_imp .* s_q + m_q ;
% 
% R2_ann_imp = 1 - sum( (q( 2 : end ) - q_ann_imp( 2 : end )).^2 ) / sum( (q( 2 : end )-m_q(2:end) ).^2 )
% 
% % ANN - improper model validation
% Xv = [ xv(1:end-1) uv(2:end) ]' ;
% Y_v = sim( net_imp, Xv ) ;
% Y_impv = [ x(1); Y_v' ] ;
% n_v__pro = Y_impv .* s_qv + m_qv ;
% 
% R2_ann_impv = 1 - sum( (qv( 2 : end ) - n_v__pro( 2 : end )).^2 ) / sum( (qv( 2 : end )-m_qv(2:end) ).^2 )
% 
% 
% % multiple ANN calibrations, saving the best model as net_opt
% N_runs = 10 ;
% R2_i = zeros(N_runs,1);
% for i = 1:N_runs
%     net_i = newff(X,Y_c,3) ; % initialization of ANN
%     net_i = train( net_i, X, Y_c ) ;
%     Y_ = sim( net_i, X ) ;
%     Y_ = [ x(1); Y_' ] ;
%     q_ann_i = Y_ .* s_q + m_q ;
%     
%     R2_i(i) = 1 - sum( (q( 2 : end ) - q_ann_i( 2 : end )).^2 ) / sum( (q( 2 : end )-m_q(2:end) ).^2 );
%     if R2_i(i) >= max(R2_i)
%         net_opt = net_i ;
%     end
% end
% 
% 
% N_runs = 10 ;
% R2_i_pro = zeros(N_runs,1);
% for i = 1:N_runs
%     net_i = newff(X,Y_c,3) ; % initialization of ANN
%     net_i = train( net_i, X, Y_c ) ;
%     Y_v = sim( net_i, Xv ) ;
%     Y_impv = [ x(1); Y_v' ] ;
%     n_v__pro = Y_impv .* s_qv + m_qv ;
%     
%     R2_i_pro(i) = 1 - sum( (qv( 2 : end ) - n_v__pro( 2 : end )).^2 ) / sum( (qv( 2 : end )-m_qv(2:end) ).^2 );
%     if R2_i_pro(i) >= max(R2_i_pro)
%         net_opt = net_i ;
%     end
% end

end