function [t_AB, t_AC, t_AD] = func_PD_Simulation(receiver_A, receiver_B, receiver_C, receiver_D, PD_coords)

    % all functon inputs are 3D coordinates (3 x 1 column matrices)
    % returns the TDOA between receivers AB, AC and AD
    % positive t_AB corresponds to greater distance to B than to A

    t_AB = (norm(receiver_B - PD_coords) - norm(receiver_A - PD_coords))/(3e8);
    t_AC = (norm(receiver_C - PD_coords) - norm(receiver_A - PD_coords))/(3e8);
    t_AD = (norm(receiver_D - PD_coords) - norm(receiver_A - PD_coords))/(3e8);

end