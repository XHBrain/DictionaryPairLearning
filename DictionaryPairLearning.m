function [P D] = DictionaryPairLearning(X, lambda, tau, m, gamma)

[dimension, num_sample, K] = size(X);

P = rand(m, dimension, K);
D = rand(dimension, m, K);
A = zeros(m, num_sample,K);

%Two norm of less than or equal to 1
D = atomLessThanOne(D);

%Generates a matrix that's inverse matrix, when Optimize P
for i = 1 : K
    X_complementary = X;
    X_complementary(:,:,i) = [];
    X_complementary = reshape(X_complementary, dimension, []);
    update_P_inv_matrix(:,:,i) = ( tau*X(:,:,i)*X(:,:,i)' + lambda*X_complementary*X_complementary' ...
                                  + gamma*eye(dimension) )^-1;
end

%i represents the iteration number, k represents the number of classification
for i = 1 : 20
    for k = 1 : K
        A(:,:,k) = ( D(:,:,k)'*D(:,:,k) + tau*eye(m) )^-1 *  ...
            ( tau*P(:,:,k)*X(:,:,k) + D(:,:,k)'*X(:,:,k) );
        
        P(:,:,k) = tau*A(:,:,k)*X(:,:,k)'*update_P_inv_matrix(:,:,k);
        
        D(:,:,k) = ADMM(D(:,:,k), X(:,:,k), A(:,:,k));
   end
end

end

%% ==========================================================================
function D = ADMM(D,X,A)

[dimension, m] = size(D);

rho = 100;
step = 0.00001;

S = D;

T = zeros(size(D));

for j = 1:50
%      sum(sum((X-D*A).^2))
%      sum(sum((D-S).^2))

    S = atomLessThanOne(S);

    temp = X-D*A;
    for i = 1:m
        temp1(:,i) = sum( temp.*(-repmat(A(i,:),dimension,1)) , 2);
    end
    D = D - step*(2*temp1 + 2*rho*(D-S+T));
    
    S = S - step*(-2*rho*(D-S+T));
    
    T = T + D - S;
end

end