function optimal_pK(f::Real,N::Real,L::Integer)

    if f == 1.0
        @warn("Only f = 1.0 is currently supported for the optimal pK")
    end
    #Optimized values for 
    if N == 1e9
        # for distance
        # optimal_pK = [0.9,  0.9, 0.89, 0.88, 0.87, 0.86,#25
                            # 0.85, 0.84, 0.83, 0.82, 0.8,#50
                            # 0.79, 0.77, 0.76, 0.74, 0.72,#75
                            # 0.7, 0.68, 0.66]
        if f==1.16
            optimal_pK = [0.9, 0.89, 0.87, 0.85, 0.83,
                               0.8, 0.77, 0.73, 0.7, 
                               0.61, 0.64, 0.54, 0.51,
                               0.38, 0.34, 0.19, 0.1,
                                0.12, 0.11, 0.12, 0.15,
                                0.1, 0.14, 0.21] 
        elseif f==1.1
            optimal_pK = [0.9, 0.89, 0.87, 0.85, 0.83, 
                                0.8, 0.78, 0.75, 0.7, 
                                0.66, 0.61, 0.55, 0.48, 
                                0.4, 0.32, 0.21, 0.09,
                                0.07, 0.07]
        end
        return L < 48 ? optimal_pK[Int(L/2)+1] : optimal_pK[end]
    elseif N == 1e8
        optimal_pK = 1
    else # Default to N = 1e7
        optimal_pK = 1
    end
end
