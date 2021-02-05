clc;
close all;
clear ;

global Gabcikovo;

streams = readtable('2. Gabcikovo.csv');
Gabcikovo.lat = 47.880090;
Gabcikovo.lon = 17.538510;
Gabcikovo.area = 133257.800000; %km2
Gabcikovo.streams = streams;

%now we can start
addpath('support functions/');

%plots 
figure;
plot(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_);
ylabel('Streamflow [m^3/s]');
xlabel('Time [days]');
title('Average Daily Streamflow');

figure;
plot(Gabcikovo.streams.AverageDailyPrecipitation_mm_d_);
ylabel('Precipitations [mm/d]');
xlabel('Time [days]');
title('Average Daily Precipitation');

figure;
plot(Gabcikovo.streams.AverageDailyTemperature___C_);
ylabel('Temperatures [°C]');
xlabel('Time [days]');
title('Average Daily Temperature');

%reshape data in 365 x "number of years" 
streamflow = reshape(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_, 365, []); 
figure;
plot(streamflow);
ylabel('Streamflow [m^3/s]');
xlabel('Time [days]');
title('Average Daily Streamflow through the years');
%COMMENT
%looking through the years maybe peaks are shifting, maybe grouping years
%together can help, there are 27 lines and then 27 colors and also I don't
%know which is the starting color


%% 
close all;

%infos 
%all the costants for time will be here in T.something
global T;
T.days = 365;

horizon.days = length(Gabcikovo.streams.Day)
%then variables are daysx1 long
horizon.years = horizon.days/T.days
min_stream = min(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_)
max_stream = max(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_)
mean_stream = mean(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_)
var_stream = var(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_)
range_stream = max_stream - min_stream

T.time = repmat( (1:365)' , horizon.years, 1 ) ;
%reshape data in 365 x "number of years" 

%Moving average and variance for different windows
%    idx    1  2   3   4
semi_amp = [1, 2, 5, 10];  % f days before and f days after
m = zeros(length(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_) , length(semi_amp) );
mi = zeros(T.days , length(semi_amp) ); 

s = zeros(length(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_) , length(semi_amp) );
s2 = zeros(length(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_) , length(semi_amp) );
sigma = zeros(T.days , length(semi_amp) );
sigma2 = zeros(T.days , length(semi_amp) );

figure;
tiledlayout(length(semi_amp)/2, 2);

for idx =1:length(semi_amp)
    %m average
    [ mi(:, idx) , m(:, idx) ] = moving_average( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ , T.days , semi_amp(idx) ) ;  
    
    %m dev 
    [ sigma2(:, idx) , s2(:, idx) ] = moving_average( ( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ - m(:,idx) ).^2 , T.days , semi_amp(idx) ) ;
    sigma           = sigma2 .^ 0.5                        ;
    s               = s2 .^ 0.5                            ;
    
    %plotting all the case for report
    nexttile;
    plot(T.time, Gabcikovo.streams.AverageDailyStreamflow_m_3_s_, '.');
    hold on;
    ylabel('Streamflow [m^3/s]');
    xlabel('Time [days]');
    plot( 1:T.days , mi(:,idx) , 'r', 'LineWidth',2  );
    legend('Observed', strcat('cs mean-', num2str( 2*semi_amp(idx) ) ) );
    title(strcat('amplitude ', num2str( 2*semi_amp(idx) ) ) );
    hold off;

end

% autocorrelation 
%normal
figure ; correlogram( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ , Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ , 30 ) ; xlabel('k') ; ylabel('r_k')

%here you select which window size you prefer based on correlogram
w_idx = 2;
%    idx    1  2   3   4
%semi_amp = [1, 5, 10, 20];  % f days before and f days after

T.f = semi_amp(w_idx); %fix the semi amplitude for next evaluations

mi_stream = mi(:,w_idx);
m_stream = m(:, w_idx);
sigma_stream = sigma(:,w_idx);
s_stream = s(:, w_idx);
x_stream = (Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ - m_stream)./s_stream;

% desonalized
figure ; correlogram( x_stream , x_stream , 30 ) ; xlabel('k') ; ylabel('r_k') ; 
title('desonalized corr');

%cross correlation
[u_prec, m_prec, s_prec] = detrend(Gabcikovo.streams.AverageDailyPrecipitation_mm_d_, T.days, T.f);

[u_temp, m_temp, s_temp] = detrend(Gabcikovo.streams.AverageDailyTemperature___C_, T.days, T.f);

figure ; correlogram( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ , Gabcikovo.streams.AverageDailyPrecipitation_mm_d_ , 10 ) ; xlabel('k') ; ylabel('r_k') ; 
title('precipitation vs stream');
figure ; correlogram( x_stream , u_prec , 10 ) ; xlabel('k') ; ylabel('r_k') ; 
title('deseasonalized precipitation vs stream');

%REMARK for time = 3 the correlation is already 0, so it makes sense not to
%use a lot of days

figure ; correlogram( Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ , Gabcikovo.streams.AverageDailyTemperature___C_ , 10 ) ; xlabel('k') ; ylabel('r_k') ; 
title('temp vs stream');
figure ; correlogram( x_stream , u_temp , 10 ) ; xlabel('k') ; ylabel('r_k') ; 
title('deseasonalized temp vs stream');

%REMARK correlation is basically always zero, so I'll says it doesn't
%depend on temperature and also on precip

%% Starting evaluation of different models
%AR p
%AR eX p q 
%ANN
close all;

%AR(p)
n = Gabcikovo.streams.AverageDailyStreamflow_m_3_s_;

%1 define J
% MSE
% coefficient of determination (Rt2)
% R2 = 1 - sum( (n - n_).^2 ) / sum( (n - m).^2 )


%2 MOdel structure M = AR(p)
%for increasing p from 1 until J decreases
imp = 1;
p = 1;
p_max = 10; %to avoid useless computations  


while imp == 1
    %3 split data set
    k_val = 9; %cross K-fold cross validation, 3 years x 9 time
    J_c_k = zeros(k_val, 1);
    J_v_k = zeros(k_val, 1);
    R2_c_k = zeros(k_val, 1);
    R2_v_k = zeros(k_val, 1);
    for k = 1:k_val
       [n_c, n_v] = k_cross( n, k_val, k);

        %4 evaluate calibration and validation data
       [J_c_k(k), R2_c_k(k), J_v_k(k), R2_v_k(k)] = ar_p(n_c, n_v, p);
    end

    %average of the k sample
    J_c_ar(p) = mean( J_c_k );
    R2_c_ar(p) = mean( R2_c_k );
    
    J_v_ar(p) = mean( J_v_k );
    R2_v_ar(p) = mean( R2_v_k );

    %is O.F. on validation worst then previous one? quit
    if ( (p > 1 && R2_v_ar(p) <= R2_v_ar(p-1) ) || p >= p_max )
        imp = 0;
          %optional
           %re-evaluate param on full dataset to find the right param and save
            %them
    end
    
    p = p+1;
    %remember best p will be final value of p -2 
end

p_opt = p -2;

%now you can plot graphs of the error for increasing p 
figure;
plot( 1:p-1, R2_c_ar, 'r');
hold on;
plot( 1:p-1, R2_v_ar, 'b');
title('R2 of AR model with increasing order');
ylabel('R2');
xlabel('order of the model');
%you could also plot J
figure;
plot( 1:p-1, J_c_ar, 'r');
hold on;
plot( 1:p-1, J_v_ar, 'b');
title('MSE of AR model with increasing order');
ylabel('MSE');
xlabel('order of the model');
%% ARX 
close all;
u = [Gabcikovo.streams.AverageDailyPrecipitation_mm_d_, Gabcikovo.streams.AverageDailyTemperature___C_ ];

arx_to_test = [ 1, 1;
                1, 2;
                1, 3;
                1, 4;
                2, 1;
                4, 1;
                p_opt, 1;
                p_opt, 2;
                p_opt, 3;
                p_opt+3, 1;
                p_opt+3, 2;
                p_opt+10, 1;
                p_opt+10, 2];

    numb_of_test = size(arx_to_test);
    numb_of_test = numb_of_test(1);
    
    %first using only precipitation
    for i = 1:numb_of_test
        %3 split data set
        k_val = 9;
        R2_v_k = zeros(k_val, 1);
        for k = 1:k_val
            [n_c, n_v] = k_cross( n, k_val, k);
            [u_c, u_v] = k_cross( u, k_val, k);
            
            %4 evaluate calibration and validation data
            p = arx_to_test(i, 1);
            q = arx_to_test(i, 2);
            [R2_c_k(k), R2_v_k(k)] = arx_p_q(n_c, u_c(:,1), n_v, u_v(:,1), p, q, 'proper');
        end
        
        %average of the k sample
        R2_c_arx_1(i) = mean( R2_c_k );
        R2_v_arx_1(i) = mean( R2_v_k );
    end
    
    %second using precipitation and also temperature
     for i = 1:numb_of_test
        %3 split data set
        k_val = 9;
        R2_v_k = zeros(k_val, 1);
        for k = 1:k_val
            [n_c, n_v] = k_cross( n, k_val, k);
            [u_c, u_v] = k_cross( u, k_val, k);
            
            %4 evaluate calibration and validation data
            p = arx_to_test(i, 1);
            q = arx_to_test(i, 2);
            [R2_c_k(k), R2_v_k(k)] = arx_p_q(n_c, u_c, n_v, u_v, p, q, 'proper' );
        end
        
        %average of the k sample
        R2_c_arx_2(i) = mean( R2_c_k );
        R2_v_arx_2(i) = mean( R2_v_k );
     end
    
    [~, I] = max(R2_v_arx_2);
    p_q_opt = arx_to_test(I,:);
    
figure;
plot( 1:length(R2_c_arx_1), R2_c_arx_1, '*-r');
hold on;
plot( 1:length(R2_c_arx_1), R2_v_arx_1, 'o-b');
plot( 1:length(R2_c_arx_1), R2_c_arx_2, '*-g');
plot( 1:length(R2_c_arx_1), R2_v_arx_2, 'o-y');
legend('calib inp:prec', 'valid inp: prec', 'calib inp: prec+temp', 'valid inp: prec +temp'); 
title('R2 of ARX model with different p and q');
ylabel('R2');
xlabel('index of test array');
%I see there is a small change so I think is because temperature doesn't
%affect the system really much.
    
%in the end using only temperature 
    for i = 1:numb_of_test
        %3 split data set
        k_val = 9;
        R2_v_k = zeros(k_val, 1);
        for k = 1:k_val
            [n_c, n_v] = k_cross( n, k_val, k);
            [u_c, u_v] = k_cross( u, k_val, k);
            
            %4 evaluate calibration and validation data
            p = arx_to_test(i, 1);
            q = arx_to_test(i, 2);
            [R2_c_k(k), R2_v_k(k)] = arx_p_q(n_c, u_c(:,2), n_v, u_v(:,2), p, q, 'proper' );
        end
        
        %average of the k sample
        R2_c_arx_3(i) = mean( R2_c_k );
        R2_v_arx_3(i) = mean( R2_v_k );
    end
    
figure;
hold on;
plot( 1:length(R2_c_arx_1), R2_v_arx_1, 'o-b');
plot( 1:length(R2_c_arx_1), R2_v_arx_3, 'o-y');
legend( 'valid inp: prec', 'valid inp: temp'); 
title('R2 of ARX model with different exo input');
ylabel('R2');
xlabel('index of test array');
%we can see how using precipitation the model is more accurate (blue line
%is over )

%now since the best model was arx(p, q) i run it in improper model
    for i = 1:numb_of_test
        %3 split data set
        k_val = 9;
        R2_v_k = zeros(k_val, 1);
        for k = 1:k_val
            [n_c, n_v] = k_cross( n, k_val, k);
            [u_c, u_v] = k_cross( u, k_val, k);
            
            %4 evaluate calibration and validation data
            p = arx_to_test(i, 1);
            q = arx_to_test(i, 2);
            [R2_c_k(k), R2_v_k(k)] = arx_p_q(n_c, u_c(:,1), n_v, u_v(:,1), p, q, 'improper');
        end
        
        %average of the k sample
        R2_c_arx_4(i) = mean( R2_c_k );
        R2_v_arx_4(i) = mean( R2_v_k );
    end
    
figure;
hold on;
plot( 1:length(R2_c_arx_1), R2_v_arx_1, 'o-b');
plot( 1:length(R2_c_arx_1), R2_v_arx_4, 'o-r');
legend( 'valid prec proper', 'valid prec improper'); 
title('R2 of ARX model with proper and improper method');
ylabel('R2');
xlabel('index of test array');
%% ANN
close all;
%2 MOdel structure M = ANN(p)
%p is the number of neuron, starts from number of input +1
%also thanks to the experience of arx model I will use only one input, i.e.
%precipitation

%for increasing p from 1 until J decreases
imp = 1;
neu_min = 3;
neu = neu_min;
neu_max = 10;

while imp == 1
    
       [R2_ann(neu+1 - neu_min), ~] = ann_p(n, u(:,1), neu);
   
    %is O.F. on validation worst then previous one? quit
    if ( (neu > neu_min && R2_ann(neu+1 - neu_min) <= R2_ann(neu - neu_min ) + 0.0005 ) || neu > neu_max )
        imp = 0;
    end
    
    neu = neu+1;
    %remember best neuron value will be final value of neuron -1 
end

neu_opt = neu - 1 - neu_min;
%now you can plot graphs of the error for increasing p 
figure;
plot( 1:length(R2_ann), R2_ann, 'o-r');
ylabel('R2');
xlabel( strcat('# of neurons - ', num2str(neu_min) ) );
ylim( [0.96 0.97]);



%% confrontations beetween the 3 models
close all;
figure;
hold on;
plot(1, R2_c_ar(p_opt), '*-b');
plot(1, R2_v_ar(p_opt), 'o-b');
plot(2, R2_c_arx_1(I), '*-r');
plot(2, R2_v_arx_1(I), 'o-r');
plot(3, R2_c_arx_4(I), '*-y');
plot(3, R2_v_arx_4(I), 'o-y');
plot(4, R2_ann(neu_opt), 'o-g');
xlim( [0 5]);
ylim( [0.95 1]);
ylabel('R2');
title('best R2 of different models');
legend('ar cal', 'ar val', 'arx prec proper cal', 'arx prec proper val', 'arx prec imp cal', 'arx prec imp val', 'ann prec'); 

%% Let's start second part 

% INDICATORS FOR WATER SUPPLY
w = prctile(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_, 45) ;
r = Gabcikovo.streams.AverageDailyStreamflow_m_3_s_;
Ny = length(T.time) / T.days;

I0 = evaluate_indic(r, w, 0, Ny, 0);

%% Start project of the dam
close all;

%move to monthly
qMonth = dailyToMonthly(Gabcikovo.streams.AverageDailyStreamflow_m_3_s_, Ny);


temp = reshape(qMonth, Ny*12, 1);
figure;
plot(1:length(temp), temp, 'b');
hold on;
plot(1:length(temp), w*ones(length(temp), 1), 'r');
xlabel('month'); ylabel('stream [m^3/month]');

% Sequent Peak Analysis
deltaT = 3600*24*[31 28 31 30 31 30 31 31 30 31 30 31]';
Q = qMonth(:).*repmat(deltaT,Ny,1) ; % m3/month
W = w*ones(size(Q)).*repmat(deltaT,Ny,1) ; % m3/month
K = zeros(size(Q));
K(1) = 0;

for t = 1:length(Q)
    K(t+1) = K(t) + W(t) - Q(t) ;
    if K(t+1) < 0
        K(t+1) = 0 ;
    end        
end

figure; plot( K )
Kopt = max(K);
param.nat.S = Gabcikovo.area * 10^6; %
hmax = Kopt/param.nat.S; 
%hmax = 1;
rmax = prctile(r , 99);
param.nat.h0 = 0;
param.nat.m = rmax/(hmax - param.nat.h0);

h_flo = hmax*1.02;

%% 
% regulated level-discharge relationship:
param.reg.w = w ;
param.reg.h_min = 0.02 ;
param.reg.h_max = hmax ;
param.reg.h1 = 0.04 ;
param.reg.h2 = 0.05 ;
param.reg.m1 = 10000 ;
param.reg.m2 = 80000 ;

h_test = [ 0 : 0.012 :  0.1] ;
r_test = regulated_release( param, h_test ) ;
figure; plot(h_test,r_test, 'o')
grid on

n = [ nan; Gabcikovo.streams.AverageDailyStreamflow_m_3_s_ ] ; % for time convention
h_init = param.reg.h_max/2 ; % initial condition

[s_nat, h_nat, r_nat] = simulate_dam( n, h_init, param, 'nat' ) ;

figure; plot( h_nat );
hold on;
plot(1:length(h_nat), param.reg.h_max*ones(length(h_nat), 1), 'r');

figure;
plot(r_nat, 'b');hold on;
plot(n(2:end), 'r');
%% let's try with some alternatives
close all
%struct of the alternative
%h1
%h2
%m1
%m2
%indicators per flood 
%indicators per l'altr
%indicators per ultimo

alternatives = [w/param.nat.m,  0.0315,         w/param.nat.m*1.3;
                hmax*0.9,       0.052,          hmax*0.5;
                param.nat.m*2,  param.nat.m,    param.nat.m/2;
                400000,         200000,         250000];
            
 [~, tries ] = size(alternatives);
 A = [I0, I0, I0];
 
 for idx = 1:tries
    param.reg.h1 = alternatives(1, idx) ;
    param.reg.h2 = alternatives(2, idx) ;
    param.reg.m1 = alternatives(3, idx) ;
    param.reg.m2 = alternatives(4, idx) ;

    
    [s_reg, h_reg, r_reg] = simulate_dam( n, h_init, param, 'reg' ) ;
% 
%     figure; plot( h_reg );
%     hold on;
%     plot(1:length(h_reg), param.reg.h_max*ones(length(h_reg), 1), 'r');
% 
    figure;
    plot(r_reg, 'b');hold on;
    plot(1:length(r_reg), w*ones(length(r_reg), 1 ), 'r');
   
%     h_test = [ -0.1 : 0.01 :  0.17] ;
% r_test = regulated_release( param, h_test ) ;
% figure; plot(h_test,r_test, 'o')
% grid on
% 
%     
    A(idx) = evaluate_indic(r_reg, w, h_reg, Ny, param.reg.h_max);
 end
 
for idx = 1:tries 
   %plot the policy (alternatives)
    %h_test = [ 0: 0.001 :  0.1] ;
    
    param.reg.h1 = alternatives(1, idx) ;
    param.reg.h2 = alternatives(2, idx) ;
    param.reg.m1 = alternatives(3, idx) ;
    param.reg.m2 = alternatives(4, idx) ;

    %r_test = regulated_release( param, h_test ) ;
    figure; 
    xx = [param.reg.h_min  param.reg.h1];
    yy = w + (xx - param.reg.h1)*param.reg.m1;
    %plot(h_test,r_test);
    line( xx , yy, 'Color', 'blue');
    hold on;
    grid on;
    xx = [param.reg.h1  param.reg.h2];
    yy =  w*ones(length(xx), 1) ;
    line( xx , yy, 'Color', 'blue');
    intersect = (w -param.reg.h2*param.reg.m2) / (param.nat.m - param.reg.m2);
    xx = [param.reg.h2  min(param.reg.h_max, intersect) ];
    yy = w + (xx - param.reg.h2)*param.reg.m2;
    line( xx , yy, 'Color', 'blue');
    plot([param.nat.h0  param.reg.h_max], [param.nat.h0  param.reg.h_max]*param.nat.m, 'r');
    xline(param.reg.h_max);
%    plot([param.nat.h0  param.reg.h_max], w*ones(length([param.nat.h0  param.reg.h_max]), 1));
    
    xlabel('level'); ylabel('release');
end

%% 
close all;
% NSGAII-EMODPS optimization

global opt_inputs;
opt_inputs.n = n ;
opt_inputs.h_init = h_init ;
opt_inputs.param = param ;
opt_inputs.h_flo = h_flo ;
opt_inputs.Ny = Ny;

addpath('support functions/NSGA2')
pop = 30 ;
gen = 25 ;
M = 2 ;
V = 4 ;

min_range = [ w/param.nat.m        hmax*0.5 	param.nat.m/2   200000 ] ;
max_range = [ w/param.nat.m*1.3    hmax*0.9    param.nat.m*2   400000 ] ;
[ chromosome_0, chromosome15 ] = nsga_2(pop,gen,M,V,min_range,max_range) ;

figure; plot( chromosome_0(:,end), chromosome_0(:,end-1), 'o') ; 
hold on; plot( chromosome15(:,6), chromosome15(:,5), 'ro')
xlabel('number of flood per year');ylabel('irrigation deficit');

% decision space
h_test = [ 0 : 0.001 :  0.1] ;
clear r_test;
for i = 1:pop
    xi = chromosome15(i,1:4);
    param.reg.h1 = xi(1);
    param.reg.h2 = xi(2);
    param.reg.m1 = xi(3);
    param.reg.m2 = xi(4);
    
    r_test(i,:) = regulated_release( param, h_test ) ;
end
figure;
plot( h_test, r_test );
hold on;
plot([param.nat.h0  param.reg.h_max], [param.nat.h0  param.reg.h_max]*param.nat.m, 'r');
xline(param.reg.h_max);
%figure; plot( h_test, r_test );
xlabel('level'); ylabel('release');


%% 

alternatives = chromosome15(1:3, 1:4)';
            
 [~, tries ] = size(alternatives);
 A = [I0, I0, I0];
 
 for idx = 1:tries
    param.reg.h1 = alternatives(1, idx) ;
    param.reg.h2 = alternatives(2, idx) ;
    param.reg.m1 = alternatives(3, idx) ;
    param.reg.m2 = alternatives(4, idx) ;

    
    [s_reg, h_reg, r_reg] = simulate_dam( n, h_init, param, 'reg' ) ;
% 
%     figure; plot( h_reg );
%     hold on;
%     plot(1:length(h_reg), param.reg.h_max*ones(length(h_reg), 1), 'r');
% 
    figure;
    plot(r_reg, 'b');hold on;
    plot(1:length(r_reg), w*ones(length(r_reg), 1 ), 'r');
   
%     h_test = [ -0.1 : 0.01 :  0.17] ;
% r_test = regulated_release( param, h_test ) ;
% figure; plot(h_test,r_test, 'o')
% grid on
% 
%     
    A_EMO(idx) = evaluate_indic(r_reg, w, h_reg, Ny, param.reg.h_max);
 end
 
 
