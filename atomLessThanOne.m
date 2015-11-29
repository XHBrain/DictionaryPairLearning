function D = atomLessThanOne(D)
% each column vectors is summation greater than 1  are unitized
A_sum = sum(D.^2,1);

index = A_sum>1.0;

for i = 1 : size(D,3)
if index(:,:,i)==0
    continue;
end

D(:, index(:,:,i), i) = D(:, index(:,:,i), i) ./ repmat(A_sum(1,index(:,:,i),i).^0.5, size(D,1), 1);

end
end