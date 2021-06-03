%Edit and run this routine when needed
theta
X
y

%Run fminunc
[theta, J_history] = gd_fminunc(X, y, theta, num_iters);

%Plot if needed
