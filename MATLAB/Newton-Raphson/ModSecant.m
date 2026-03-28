rcv_A = [-0.9; -0.9; 0];
rcv_B = [0.9; -0.9; 0];
rcv_C = [-0.9; 0.9; 0];
rcv_D = [0.9; 0.9; 0];
PD = [-2.1; 9; 5];

[t_AB, t_AC, t_AD] = func_PD_Simulation(rcv_A, rcv_B, rcv_C, rcv_D, PD);

d = 0.01; % delta value for distance
d_t = 1e-10; % delta value for time

f1 =  @(x,y,z,t) norm([x - rcv_A(1), y - rcv_A(2), z - rcv_A(3)]) - 3e8*t;
f2 =  @(x,y,z,t) norm([x - rcv_B(1), y - rcv_B(2), z - rcv_B(3)]) - 3e8*(t+t_AB);
f3 =  @(x,y,z,t) norm([x - rcv_C(1), y - rcv_C(2), z - rcv_C(3)]) - 3e8*(t+t_AC);
f4 =  @(x,y,z,t) norm([x - rcv_D(1), y - rcv_D(2), z - rcv_D(3)]) - 3e8*(t+t_AD);

J = @(x,y,z,t) [((f1(x+d,y,z,t)-f1(x,y,z,t))/(d)) ((f1(x,y+d,z,t)-f1(x,y,z,t))/(d)) ((f1(x,y,z+d,t)-f1(x,y,z,t))/(d)) ((f1(x,y,z,t+d_t)-f1(x,y,z,t))/(d_t)); ...
    ((f2(x+d,y,z,t)-f2(x,y,z,t))/(d)) ((f2(x,y+d,z,t)-f2(x,y,z,t))/(d)) ((f2(x,y,z+d,t)-f2(x,y,z,t))/(d)) ((f2(x,y,z,t+d_t)-f2(x,y,z,t))/(d_t)); ...
    ((f3(x+d,y,z,t)-f3(x,y,z,t))/(d)) ((f3(x,y+d,z,t)-f3(x,y,z,t))/(d)) ((f3(x,y,z+d,t)-f3(x,y,z,t))/(d)) ((f3(x,y,z,t+d_t)-f3(x,y,z,t))/(d_t)); ...
    ((f4(x+d,y,z,t)-f4(x,y,z,t))/(d)) ((f4(x,y+d,z,t)-f4(x,y,z,t))/(d)) ((f4(x,y,z+d,t)-f4(x,y,z,t))/(d)) ((f4(x,y,z,t+d_t)-f4(x,y,z,t))/(d_t))];

r = [0; 0; 0; 0]; % initial result
Na = 0.0001;
check = abs(f1(r(1),r(2),r(3),r(4))) + abs(f2(r(1),r(2),r(3),r(4))) + abs(f3(r(1),r(2),r(3),r(4))) + abs(f4(r(1),r(2),r(3),r(4)));
prm = 0; % parametric variable used to generate coordinates along a multidimensional spiral


while ((abs(check) > Na)|| isnan(check)) && (prm <= 100) % isnan to see if a singular matrix occurred. NaN would cause abs(check)>Na to be false
    r(1) = prm.*cos(prm);
    r(2) = prm.*sin(prm);
    r(3) = prm.*cos(prm);
    r(4) = (prm.*sin(prm))/1e8;
    check = inf;
    
    iter_modsec = 0; % mod secant iterator

    while (abs(check) > Na) &&  (iter_modsec <= 10000)
        r_0 = r;
        J_0 = J(r_0(1),r_0(2),r_0(3),r_0(4));
        f_0 = [f1(r_0(1),r_0(2),r_0(3),r_0(4)); f2(r_0(1),r_0(2),r_0(3),r_0(4)); f3(r_0(1),r_0(2),r_0(3),r_0(4)); f4(r_0(1),r_0(2),r_0(3),r_0(4))];

        % rcondJ = rcond(J_0);
        % if rcondJ < 1e-12
        %     break;  % skip to next iteration instead of crashing
        % end

        r = r_0 - J_0\f_0;
        % a singular J_0 matrix would cause matlab to break out of this
        % loop and check to become NaN
    
        check = abs(f1(r(1),r(2),r(3),r(4))) + abs(f2(r(1),r(2),r(3),r(4))) + abs(f3(r(1),r(2),r(3),r(4))) + abs(f4(r(1),r(2),r(3),r(4)));
        iter_modsec = iter_modsec+1;
    
    end

    prm = prm + 0.1;

end