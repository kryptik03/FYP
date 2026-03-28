function [true_source_location, TDOA_N] = func_Sorting_data_predict(data_predict, data_set)

    % Define the PD coordinates
    pointX = data_predict(data_set, 1);    pointY = data_predict(data_set, 2);    pointZ = data_predict(data_set, 3);
    true_source_location = [pointX, pointY, pointZ];

    % TDOA_N = [t21, t31, t41];
    TDOA_N = [data_predict(data_set, 4), data_predict(data_set, 5), data_predict(data_set, 6)];

end