function [r1, r2, r3, d1, d2, d3, d4] = func_Objective_Function(x0, y0, z0, receiver1, receiver2, receiver3, receiver4, speed_of_EM, TDOA_N)

    % Calculate the residuals for each TDOA measurement
    d1 = sqrt((x0 - receiver1(1))^2 + (y0 - receiver1(2))^2 + (z0 - receiver1(3))^2);
    d2 = sqrt((x0 - receiver2(1))^2 + (y0 - receiver2(2))^2 + (z0 - receiver2(3))^2);
    d3 = sqrt((x0 - receiver3(1))^2 + (y0 - receiver3(2))^2 + (z0 - receiver3(3))^2);
    d4 = sqrt((x0 - receiver4(1))^2 + (y0 - receiver4(2))^2 + (z0 - receiver4(3))^2);

    % Add the error compensation term
    r1 = d2 - d1 - ((speed_of_EM) * TDOA_N(1));
    r2 = d3 - d1 - ((speed_of_EM) * TDOA_N(2));
    r3 = d4 - d1 - ((speed_of_EM) * TDOA_N(3));
    
end