function xMonth = dailyToMonthly( x, N )

% function qMonth = dailyToMonthly( x, N )
%
% function to convert a daily timeseries into monthly average values
% input:    - x = trajectory of daily values (vector of 365*N elements)
%           - N = number of years
%
% output:   xMonth = monthly average values (matrix 12*N)

% reshape vector into matrix
xx = reshape(x,365,N);
% id of months
dm = [1*ones(31,1); 2*ones(28,1); 3*ones(31,1); 4*ones(30,1); 5*ones(31,1); 6*ones(30,1); 7*ones(31,1);...
    8*ones(31,1); 9*ones(30,1); 10*ones(31,1); 11*ones(30,1); 12*ones(31,1)] ;
% monthly average
xMonth = nan(12,N);
for i = 1:12
    for j = 1:N
        idx = dm == i;
        x_ = xx(idx,j) ;
        xMonth(i,j) = mean( x_ ) ; 
        clear x_
    end
end

end
