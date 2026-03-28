function [train_location_coordinates, train_TDOA_data, val_location_coordinates, val_TDOA_data] = func_load_data(matFileName_Train_Coordinates, matFileName_Train_TDOA, matFileName_Val_Coordinates, matFileName_Val_TDOA)
    % Load data from the .mat files
    trainData_Coordinates = load(matFileName_Train_Coordinates);
    trainData_TDOA = load(matFileName_Train_TDOA);
    valData_Coordinates = load(matFileName_Val_Coordinates);
    valData_TDOA = load(matFileName_Val_TDOA);

    % Extract variables from the loaded data
    train_location_coordinates = trainData_Coordinates.train_location_coordinates;
    train_TDOA_data = trainData_TDOA.train_TDOA_data;
    val_location_coordinates = valData_Coordinates.val_location_coordinates;
    val_TDOA_data = valData_TDOA.val_TDOA_data;
end