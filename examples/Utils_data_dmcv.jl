
function optimal_amp(f::Real,N::Real,D::Integer)

    if f != 1.0
        @warn("Only f = 1.0 is currently supported for the optimal amplitudes")
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
    # if N == 1e10
    #     optimal_amp =
    # elseif N == 1e9
    #     optimal_amp =
    # elseif N == 1e8
    #     optimal_amp =
    # else # Default to N = 5e7
    #     optimal_amp =
    return D < length(optimal_amp) ? optimal_amp[D+1] : optimal_amp[end]
end





function optimal_pK(f::Real,N::Real,D::Integer)

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
    else # Default to N = 1e7
        optimal_pK = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
                            0.50, 0.50, 0.45, 0.45, 0.45,
                            0.40, 0.40, 0.35, 0.33, 0.30]
    end

    return D < length(optimal_pK) ? optimal_pK[D+1] : optimal_pK[end]
end


# function optimal_renyi(f::Real,N::Real,D::Integer)
#     if f != 1.0
#         @warn("Only f = 1.0 is currently supported for the optimal Rényi parameter")
#     end

#     if N == 1e10
#         optimal_renyi =
#     elseif N == 1e9
#         optimal_renyi =
#     elseif N == 1e8
#         optimal_renyi =
#     else # Default to N = 1e7
#         optimal_renyi =
#     return D < length(optimal_renyi) ? optimal_renyi[D+1] : optimal_renyi[end]
# end