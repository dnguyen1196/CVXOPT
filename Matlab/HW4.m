
m = 30;
n = 100;
A = randn(m,n) + 1i*randn(m,n);
b = randn(m,1) + 1i*randn(m,1);

cvx_begin;
    variable x(n) complex;
    minimize( norm(x, 2 ) );
    subject to
        A * x == b;
cvx_end;
x_norm2_sol = x;

cvx_begin;
    variable xinf(n) complex;
    minimize( norm(xinf, inf ) );
    subject to
        A * xinf == b;
cvx_end;
x_norm_inf_sol = xinf;




