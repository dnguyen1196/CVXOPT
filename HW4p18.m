n=100;
m=300;
A=rand(m,n);
b=A*ones(n,1)/2;
c=-rand(n,1);

cvx_begin
    variable x(n)
    minimize dot(c,x)
    subject to
        A * x <= b;
        x >= 0; % x must be non negative?
        x <= 1;
cvx_end

T = linspace(0,1,100);
T = T';

Feas = zeros(n,3);
maxVio = zeros(numel(T),1);
objVal = zeros(numel(T),1);

colorCode = [[0 1 0];[1 0 0]];

for i = 1: numel(T) % For each cut off
   t = T(i);
   x_hat = zeros(n,1); 
   for j = 1:n % Find x_hat based on cutoff
       if x(j) >= t
           x_hat(j) = 1;
       end
   end
   
   r = A*x_hat - b;
   
   % Find max violation
   maxVio(i) = max(r);
   if maxVio(i) > 0 % Decide feasibility
      Feas(i,:) = [1 0 0]; % Mark as infeasible
      objVal(i) = Inf; % Assign objective value
   else
      Feas(i,:) = [0 0 1];
      objVal(i) = dot(c,x_hat);
   end
end



