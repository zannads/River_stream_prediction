function [x_c , x_v] = k_cross( x, k, idx)
    %cross K-fold cross validation, we use k=9 => 3 years x 9 time
    
    
        %if val is at the end
        if(idx == 1)
            x_c = x( 1 : (3*365*(k-idx) ) , : ) ;
        end
        if(idx == k )
            x_c = x( (3*365 + 1 ) : end , :) ;
        end
        %if valid is somewhere in the center
        if (idx < k && idx > 1)
            x_c = [x( 1 : (3*365*(k-idx)) , : ) ; x( (3*365*(k-idx+1) +1 ) : end , : ) ];
        end
        %anyway n_v should be
        x_v = x( (3*365*(k-idx) +1) : (3*365*(k-idx+1) ) , : );
        
end