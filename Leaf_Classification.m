% Leaf_Classification
% Eriola Hajro


% Uncomment this line when running the program for the first time
% load('LeafData.mat', 'Ptrain', 'Ptest', 'names', 'Prgb');

% size of an image
sz = size(Ptrain{1}{1});
% preallocate A
A = zeros(sz(1)*sz(2),7,10);

% creating matrix A to hold all images in Ptrain
for i=1:10 
    for j=1:7
        aj = reshape(Ptrain{i}{j},sz(1)*sz(2),1);

        A(:,j,i) = aj;
    end
end

% define initial guess for dominant eigenvector
x0 = rand(length(A),1);
% define initial guess for second most dominant eigenvector
x0_2 = rand(length(A),1);

% define parameters
imax = 1000; % max number of iterations
etol = 1e-10; % tolerance for eigenvalue change
vtol = 1e-1; % tolerance for eigenvector equation

% preallocate
mu1 = zeros(1,10);
v1 = zeros(length(A),10);
mu2 = zeros(1,10);
v2 = zeros(length(A),10);
B1 = zeros(length(A),1,10);
B2 = zeros(length(A),2,10);
E = zeros(10,4);


% call the power method for each species
for i=1:10
    [mu,v] = Power_Method_Leaf(A(:,:,i),x0,imax,etol,vtol);
    % calculate the most dominant eigenvalue for each species
    mu1(i) = mu(end);
    % most dominant eigenvector for each species
    v1(:,i) = v(:,end);

    % to be used for the plot below
    % compute norm of error at each iteration
    e = vecnorm(v(:,1:end-1)-v(:,end),1); 
    % collect values of e
    E(i,:) = [e(1:end-1) e(2:end)];

    [second_mu, second_v] = Power_Method_Leaf2(A(:,:,i),mu1(i),v1(:,i),x0_2,imax,etol,vtol);
    % calculate the second most dominant eigenvalue for each species
    mu2(i) = second_mu(end);
    % second most dominant eigenvector for each species
    v2(:,i) = second_v(:,end);

    % We create two subspaces for each species with the dominant 
    % vectors we calculated above: V1=span{v1} and V2=span{v1 v2}
    
    % We define two matrices with the vectors that serve as a basis for
    % each subspace as shown below:
    B1(:,:,i) = [v1(:,i)];
    B2(:,:,i) = [v1(:,i) v2(:,i)];

end

% UNCOMMENT THE LINES BELOW IF YOU WANT TO BUILT THE CONVERGENCE PLOT

% % fit line to convergence plot
% fit = polyfit(log10(E(:,1)),log10(E(:,2)),1);
% 
% % plot best fit line
% xfit = log10(E(:,1));
% yfit = fit(1)*xfit+fit(2);
% h = plot(xfit,yfit,'r-','linewidth',2);
% 
% % plot e_{i+1} vs. e_{i}
% hold on
% plot(log10(E(:,1)),log10(E(:,2)),'o','markersize',8,'markerfacecolor','b')
% 
% % add legend to plot with convergence value
% leg_label = ['best fit line, \alpha=' num2str(fit(1))];
% legend(h,leg_label,'location','northwest')
% 
% % add plot labels
% xlabel('e_i')
% ylabel('e_{i+1}')
% title('Order of Convergence')
% set(gca,'fontsize',20)
% set(gcf,'color','w')




% preallocating
d1 = zeros(1,10);
d2 = zeros(1,10);
f1 = zeros(30,1);
f2 = zeros(30,1);
f3 = zeros(30,1);

% creating matrix T to hold all images in Ptest
for i=1:10 
    for j=1:3
        % concatenate the columns of each image to form vector t
        t = reshape(Ptest{i}{j},sz(1)*sz(2),1);

        % compute distance between t and each subspace V1_1, V1_2,..., V1_10
        % using formula e = t-B1*(B1'*B1)^{-1}*B'*t
        % then do the same for each subspace V2_1, V2_2,...,V2_10
        for k=1:10
            B = B1(:,:,k);
            % d1(k) is the distance between t and the subspace V1 associated
            % with species of type k
            d1(k) = norm(t-(B/(B'*B)*(B'*t)));

            % now do the same for V2 
            D = B2(:,:,k);
            % d2(k) is the distance between t and the subspace V1 associated
            % with species of type k
            d2(k) = norm(t-(D/(D'*D)*(D'*t)));
        end

        % get the minimum value and the index of that min_val from d
        % the index of the min value represents the type of species for
        % which the distance between t and the subspace V1 is the smallest
        [min_val1, min_idx1] = min(d1);

        % now get the index for the minimum value using V2
        [min_val2, min_idx2] = min(d2);

        % there are 30 pictures in total in Ptest -> in order to correctly
        % save the predicted and the actual type of species for each
        % picture below,  we need each picture to be associated 
        % with an index from 1 to 30 -> this is calculated here:
        idx = 3*(i-1)+j; % so 2nd picture in species of type 3 = 3*(3-1)+2 = 8

        % for each picture in Ptest, save the type of species that it
        % actually belongs to, in this case i
        f1(idx) = i;

        % save the type of species that the picture belongs to according 
        % to our classification algorithm using subspace V1 ( the case with 
        % the most dominant eigenvector only)
        f2(idx) = min_idx1;

        % save the type of species that the picture belongs to according 
        % to our classification algorithm using subspace V2 ( the case with 
        % the two most dominant eigenvectors)
        f3(idx) = min_idx2;

    end
end

% compute confusion matrix and chart for the case when we use the most dominant eigenvector only
C1 = confusionmat(f1,f2);
confusionchart(C1);

% compute confusion matrix and chart for the case when we use the two most
% dominant eigenvectors
C2 = confusionmat(f1,f3);
confusionchart(C2);


function [mu,x] = Power_Method_Leaf(A,x0,imax,etol,vtol)
% This function applies the power method to approximate the
% dominant eigenvalue of A*A' and a corresponding eigenvector
% Inputs:
%   A - nxn matrix
%   x0 - nx1 initial guess for eigenvector
%   imax - max number of iterations
%   etol - tolerance for change in eigenvalue approx.
%   vtol - tolerance for eigenvalue equation
% Outputs:
%   mu - eigenvalue approximations
%   x - eigenvector approximations

% store initial x vector and eigenvalue approximation
x = x0;
% changed from x0'*A*x0
mu = x0'*A*(A'*x0);

% loop through imax times
for i=1:imax

    % iterate x vector
    x(:,i+1) = A*(A'*x(:,i))/norm(A*(A'*x(:,i)));
    % find eigenvalue approximation
    mu(i+1) = x(:,i+1)'*A*(A'*x(:,i+1));

    % check for convergence of eigenvalue
    if abs(mu(i+1)-mu(i)) < etol*mu(i)
        break
    end
end

% check if eigenvalue approximation converged
if abs(mu(end)-mu(end-1)) > etol*mu(end-1)
    disp('Eigenvalue approximation did not converge')
end
% check if eigenvector equation is satisfied
if norm(A*(A'*x(:,end))-mu(end)*x(:,end)) > vtol
    disp('Eigenvector equation not satisfied')
end

end


function [second_mu,second_x] = Power_Method_Leaf2(A,dom_mu,dom_v,x0,imax,etol,vtol)
% This function applies the power method to approximate the
% second most dominant eigenvalue of A*A' and a corresponding eigenvector
% Inputs:
%   A - nxn matrix
%   dom_mu - most dominant eigenvalue
%   dom_v - most dominant eigenvector
%   x0 - nx1 initial guess for eigenvector
%   imax - max number of iterations
%   etol - tolerance for change in eigenvalue approx.
%   vtol - tolerance for eigenvalue equation
% Outputs:
%   mu - second most dominant eigenvalue approximations
%   x - second most dominant eigenvector approximations

% store initial x vector and eigenvalue approximation
second_x = x0;
second_mu = x0'*(A*(A'*x0)-dom_mu*dom_v*(dom_v'*x0));

% loop through imax times
for i=1:imax

    % iterate x vector
    second_x(:,i+1) = (A*(A'*second_x(:,i))-dom_mu*dom_v*(dom_v'*second_x(:,i)))/norm(A*(A'*second_x(:,i))-dom_mu*dom_v*(dom_v'*second_x(:,i)));
    % find eigenvalue approximation
    second_mu(i+1) = second_x(:,i+1)'*(A*(A'*second_x(:,i+1))-dom_mu*dom_v*(dom_v'*second_x(:,i+1)));

    % check for convergence of eigenvalue
    if abs(second_mu(i+1)-second_mu(i)) < etol*second_mu(i)
        break
    end
end

% check if eigenvalue approximation converged
if abs(second_mu(end)-second_mu(end-1)) > etol*second_mu(end-1)
    disp('Eigenvalue approximation2 did not converge')
end
% check if eigenvector equation is satisfied
% if norm(A*second_x(:,end)-second_mu(end)*second_x(:,end)) > vtol
if norm(A*(A'*second_x(:,end))-dom_mu*dom_v*(dom_v'*second_x(:,end))-second_mu(end)*second_x(:,end)) > vtol
    disp('Eigenvector equation2 not satisfied')
end

end