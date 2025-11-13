
function optimal_amp(f::Real,L::Integer)

    if f != 1.0
        @warn("Only f = 1.0 is supported for the optimal amplitudes")
    end

    optimal_amp = [1.07, 1.05, 1.03, 1.01, 0.99, 0.96, 
                        0.93, 0.90, 0.87, 0.85, 0.85, 
                        0.85, 0.84, 0.84, 0.84, 0.84,
                        0.83, 0.83, 0.82, 0.82, 0.81,
                        0.81, 0.80, 0.80, 0.79, 0.79, 
                        0.78, 0.78, 0.77, 0.77, 0.77, 
                        0.77, 0.77, 0.76, 0.76, 0.75, 
                        0.75, 0.75, 0.74, 0.74, 0.74, 
                        0.74, 0.74, 0.74, 0.74, 0.73, 
                        0.73, 0.73, 0.73, 0.73, 0.73, 
                        0.73, 0.73, 0.73, 0.73, 0.72, 
                        0.72, 0.72, 0.72, 0.72, 0.72, 
                        0.72, 0.72, 0.72, 0.72, 0.71, 
                        0.71, 0.71, 0.71, 0.71, 0.71]
    return L < length(optimal_amp) ? optimal_amp[L+1] : optimal_amp[end]
end





function optimal_pK(f::Real,N::Real,L::Integer)

    if f != 1.0
        @warn("Only f = 1.0 is currently supported for the optimal pK")
    end

    if N == 1e10
        optimal_pK = [0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
                            0.90, 0.90, 0.90, 0.90, 0.90,
                            0.87, 0.87, 0.87, 0.87, 0.85,
                            0.85, 0.85, 0.83, 0.83, 0.83,
                            0.83, 0.83, 0.83, 0.83, 0.83,
                            0.80, 0.80, 0.80, 0.75, 0.75,
                            0.75, 0.75, 0.75, 0.70, 0.70,
                            0.70, 0.70, 0.70, 0.70, 0.70]
    elseif N == 1e9
        optimal_pK = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85,
                            0.85, 0.80, 0.80, 0.80, 0.80,
                            0.80, 0.80, 0.80, 0.75, 0.75,
                            0.75, 0.75, 0.75, 0.75, 0.75,
                            0.70, 0.70, 0.65, 0.65, 0.65,
                            0.65, 0.65, 0.65, 0.60, 0.60,
                            0.60, 0.60, 0.60, 0.55, 0.55,
                            0.55, 0.55, 0.55, 0.50, 0.50]
    elseif N == 1e8
        optimal_pK = [0.70, 0.70, 0.70, 0.70, 0.70, 0.70,
                            0.70, 0.70, 0.70, 0.65, 0.65,
                            0.65, 0.65, 0.60, 0.60, 0.60,
                            0.60, 0.60, 0.55, 0.55, 0.55,
                            0.50, 0.50, 0.45, 0.45, 0.45,
                            0.45, 0.45, 0.45, 0.45]
    elseif N == 1e7
        optimal_pK = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
                            0.50, 0.50, 0.45, 0.45, 0.45,
                            0.40, 0.40, 0.35, 0.33, 0.30]
    elseif N == 5e6
        optimal_pK = [0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
                            0.40, 0.40, 0.38, 0.38, 0.36,
                            0.32, 0.30, 0.28, 0.26, 0.24]
    else # Default to N = 3e6
        optimal_pK = [0.39, 0.39, 0.39, 0.37, 0.37, 0.37,
                            0.37, 0.36, 0.35, 0.34, 0.30,
                            0.28, 0.28]
    end

    return L < length(optimal_pK) ? optimal_pK[L+1] : optimal_pK[end]
end



function optimal_renyi(f::Real,N::Real,L::Integer)
    if f != 1.0
        @warn("Only f = 1.0 is currently supported for the optimal Rényi parameter")
    end

    if N == 1e10
        optimal_renyi = [0.6, 0.20, 0.55, 0.40, 5.58, 0.95, 
                            1.05, 0.60, 1.00, 1.00, 1.20, 
                            1.15, 1.15, 1.00, 1.15, 1.20, 
                            1.00, 1.05, 0.85, 1.00, 4.20, 
                            1.17, 1.00, 1.80, 1.00, 1.00, 
                            1.60, 1.90, 1.00, 2.60, 1.00, 
                            1.80, 2.60, 1.80, 2.60, 1.80, 
                            1.80, 1.00, 1.00, 1.00, 1.00]*1e-5
            return L < length(optimal_renyi) ? optimal_renyi[L+1] : optimal_renyi[end]
    elseif N == 1e9
        optimal_renyi = [0.95, 0.95, 1.00, 1.00, 1.00, 1.63,
                            1.55, 2.00, 2.00, 2.40, 1.00,
                            2.20, 2.00, 2.45, 2.80, 1.00,
                            23.4, 3.20, 3.20, 4.00, 3.20,
                            3.20, 4.20, 1.38, 3.80, 7.48,
                            4.15, 5.00, 5.20, 7.20, 4.20,
                            6.60, 4.20, 4.20, 4.20, 6.61, 
                            7.20, 6.61, 8.02, 7.41, 7.20]*1e-5
            return L < length(optimal_renyi) ? optimal_renyi[L+1] : optimal_renyi[end]
    elseif N == 1e8
        optimal_renyi = [4.81, 4.81, 13.8, 5.40, 6.01, 6.61,
                            8.21, 8.41, 8.41, 10.8, 11.4,
                            13.2, 11.4, 13.2, 11.4, 13.2,
                            15.0, 15.6, 18.6, 18.6, 19.1,
                            21.6, 21.6, 26.6, 26.0, 28.2,
                            25.8, 25.8, 25.8]*1e-5
            return L < length(optimal_renyi) ? optimal_renyi[L+1] : optimal_renyi[end]
    elseif N == 1e7
        optimal_renyi = [3.96, 3.96, 4.53, 4.79, 5.21, 5.29,
                            5.77, 6.13, 7.33, 8.05, 8.10,
                            9.97, 11.9, 13.1, 14.8, 15.0,
                            15.0, 15.0, 13.4, 13.4]*1e-4
            return L < length(optimal_renyi) ? optimal_renyi[L+1] : optimal_renyi[end]
    elseif N == 5e6
        optimal_renyi = [5.40, 5.40, 5.40, 6.36, 6.36, 6.98,
                            6.98, 8.11, 8.11, 10.1, 10.1, 
                            13.3, 13.3, 16.7, 16.7, 19.8]*1e-4
            return L < length(optimal_renyi) ? optimal_renyi[L+1] : optimal_renyi[end]
    elseif N == 3e6
        optimal_renyi = [0.96, 0.96, 0.96, 1.13, 1.11, 1.17,
                            1.17, 1.42, 1.42, 1.66, 1.66,
                            2.12, 2.21]*1e-3
            return L < length(optimal_renyi) ? optimal_renyi[L+1] : optimal_renyi[end]

    else # If there's no data, return 0 (i.e. perform the optimization via Optim)
        return 0.0
    end
end
