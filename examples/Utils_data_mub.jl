function optimal_pK(f::Real,N::Real,v::Real)

    if f != 1.0
        @warn("Only f = 1.0 is currently supported for the optimal pK")
    end

    if v != 0.8
        @warn("Tables are optimized only for v = 0.8")
    end

    if N == 1e10
        optimal_pK = 0.85
    elseif N == 1e9
        optimal_pK = 0.8
    elseif N == 1e8
        optimal_pK = 0.7
    else # Default to N = 1e7
        optimal_pK = 0.6
    end

    return optimal_pK
end