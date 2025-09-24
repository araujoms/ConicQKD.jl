using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia
import Hypatia.Cones
import JLD2

using Printf
using Parameters



@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    ϵCR::T = 1e-11
    ϵPA::T = 9e-11
    ϵPE::T = 9e-11
    ϵcompPE::T = 9e-11
end


@with_kw struct Finite_pars{T<:AbstractFloat}
    ϵPA::T
    ϵPE::T
    v::T
    d::T
    N::T
    pK::T
    n::T
    ϵcompPE::T
    leak_EC::T
    analytical_mub::Bool
    fast::Bool
end

function FiniteSKR(renyiα, finiteSKR_pars::Finite_pars{T}) where {T<:AbstractFloat}
    
    # unpack pars
    @unpack ϵPA, ϵPE, v, d, N, pK, n, ϵcompPE, leak_EC, analytical_mub, fast = finiteSKR_pars
    
    # Total correction
    correction = leak_EC + Finite_corrections(renyiα, ϵPE, ϵPA)/N

    # Conic program
    h_renyi = hae_mub_general(v, d, N, pK, n, ϵcompPE, renyiα; analytical_mub, fast)

    FiniteSecretKey = h_renyi - correction

    # Some log info
    @printf("α-1 = %.5e, SKR = %.2e \n", renyiα-1, FiniteSecretKey)

    return FiniteSecretKey
end


function numerical_mubs(d)
    mub_dict = JLD2.load("examples/mubs.jld2")
    return mub_dict["mubs"][d]
end

"Decoherence map acting on Alice's key storage"
function zgmap(rho::AbstractMatrix, d::Integer)
    K = zgkraus(d)
    zgrho = sum(K[i] * rho * K[i] for i ∈ 1:d)
    return Hermitian(zgrho)
end

function zgkraus(d::Integer)
    K = [kron(proj(i, d), I(d)) for i ∈ 1:d]
    return K
end

Finite_corrections(renyiα::T, ϵPE::T, ϵPA::T) where {T<:AbstractFloat} =
    (log(1/ϵPE)  + log(1/ϵPA))* renyiα/(renyiα-T(1)) - 2

function EC_cost_mub(v::T, d::Integer, f::T, N::T, pK::T, ϵCR::T) where {T<:AbstractFloat}
    
    # H(A|B)
    leak_EC = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(T(d) - 1)
    
    leak_EC *= f*pK^2                 # EC efficiency and pK
    leak_EC += ceil(log2(inv(ϵCR)))/N # Correctness cost

    return leak_EC
end


function simulated_probabilities_mub(v::T,d::Integer,pK::T,n::Integer) where {T<:AbstractFloat}
    
    p2 = ((T(1)-pK)*inv(n-1))^2
    W = v + (1 - v) / d

    # Basis coincidence
    p_sim = p2 * W * ones(n-1)

    # Anything else
    push!(p_sim, T(1) - pK^2 - (n-1)*p2*W)

    return p_sim
end

function constraint_probabilities_mub(ρ::AbstractMatrix, d::Integer, pK::T, n::Integer; analytical_mub::Bool = false) where {T<: AbstractFloat}
    if analytical_mub
        mubs = mub(Complex{T}, d) # analytical MUBs from the package Ket
    else
        mubs = numerical_mubs(d)
    end
    if T != Float64 && !analytical_mub
        @warn "To achieve higher precision analytical MUBs are needed."
    end

    # Vector of probabilities for each basis
    p   = [pK]
    pPE = [(T(1)-pK)*inv(n-1) for i ∈ 1:n-1]
    append!(p, pPE)

    """ Start of idea - A&B sum their stats when they measure on the same basis and outcomes coincide"""
    # Probability of basis coincidence
    p2 = ((T(1)-pK)*inv(n-1))^2
    b = [zeros(Complex{T}, d^2, d^2) for i ∈ 1:n]
    for i ∈ 1:n-1, j ∈ 1:d
        temp = ketbra(mubs[i+1][:, j]) # Note that we skip the first MUB
        b[i] += p2*kron(temp, transpose(temp))
    end

    # Then they sum all other cases (???)
    for i ∈ 1:n, j ∈ 1:n, k ∈ 1:d, l ∈ 1:d
        if i == 1 && j == 1
            continue
        elseif i == j && k == l # This also discards i == 0 (key)
            continue
        end
        tempA = ketbra(mubs[i][:, k])
        tempB = ketbra(mubs[j][:, l])
        b[n] += p[i]*p[j]*kron(tempA, transpose(tempB))
    end

    cleanup!.(b)
    b = Hermitian.(b)

    

    """ End of idea """

    

    # b = [zeros(Complex{T}, d^2, d^2) for i ∈ 1:n-1, j ∈ 1:n-1]
    # for i ∈ 1:n-1, j ∈ 1:n-1, k ∈ 1:d, l ∈ 1:d
        # tempA = ketbra(mubs[i][:, k])
        # tempB = ketbra(mubs[j][:, l])
        # b[i,j] += p[i]*p[j]*kron(tempA, transpose(tempB))
    # end
    # cleanup!.(b)
    # b = Hermitian.(b)
    return real(dot.(Ref(ρ), b))
end

function hae_mub_general(
    v::T, 
    d::Integer, 
    N::T, 
    pK::T, 
    n::Integer, 
    ϵcompPE::T, 
    renyiα::T; 
    analytical_mub::Bool = false, 
    fast::Bool = false
) where {T<:AbstractFloat}
    
    is_complex = true
    model = GenericModel{T}()
    hermitian_space = Ket._sdp_parameters(is_complex)[3]
    R = is_complex ? Complex{T} : T

    
    # Variables
    @variable(model, ρ[1:d^2, 1:d^2], hermitian_space)
    @variable(model, q_K ≥ 0)
    @variable(model, q[1:n] ≥ 0) 
    @variable(model, h_QKD)
    @variable(model, h_KL)


    # Simulated probabilities
    p_sim = simulated_probabilities_mub(v, d, pK, n)
    p_ρAB = constraint_probabilities_mub(ρ, d, pK, n; analytical_mub)
    
    # Constraint on states
    @constraint(model, tr(ρ)==T(1))

    # Constraints on probabilities
    @constraint(model, sum(q) + q_K == 1)

    # Constraints on exp vals via KL divergence
    @constraint(model, [h_KL; p_ρAB[:];pK^2; q[:];q_K] in Hypatia.EpiRelEntropyCone{T}(1+2+2*length(q[:]),false))

    # Finite bounds via a Bretagnolle-Huber-Carol estimator
    C_alphbet = length(q[:])+1 # Key (1) + Coincident bases (n-1) + Non-coincident (1)
    δ = sqrt((2*C_alphbet*log(2) - 2*log(ϵcompPE))/N)
    @constraint(model, [δ; q[:] - p_sim[:];q_K - pK^2] in Hypatia.EpiNormInfCone{T,T}(1+1+length(q[:]),true))

    # Key map
    Ghat = [I(d^2)]
    Zhat = zgkraus(d)
    blocks = [(i-1)*d+1:i*d for i ∈ 1:d]

    vec_dim = Cones.svec_length(R, d^2)
    ρ_vec = svec(ρ)
    

    # Conic program 
    @variable(model, u)
    if fast
        β = inv(renyiα)
        sβ = β < 1 ? -1 : 1
        @constraint(
            model,
            [u; ρ_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat, Zhat, 1 + vec_dim; blocks)
        )
    else
        β = inv(2 - inv(renyiα))
        sβ = β < 1 ? -1 : 1
        dim_σ = size(Zhat[1],2)
        @variable(model, σ[1:dim_σ, 1:dim_σ], Hermitian)
        @constraint(model, tr(σ) == 1)
        σ_vec = svec(σ)
        @constraint(
            model,
            [u; ρ_vec; σ_vec] in EpiRenyiQKDTriCone{T,Complex{T}}(β, Ghat, Zhat, 1 + 2vec_dim; blocks)
        )
    end
    @constraint(model, [h_QKD * (β - 1), 1, sβ * u] in MOI.ExponentialCone())
    @objective(model, Min, renyiα*inv(log(T(2))*(renyiα-T(1)))*h_KL + (pK-δ)*inv(log(T(2)))*h_QKD)

    # Optimize
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Extract results
    h_renyi = dual_objective_value(model)

    return h_renyi
end


"""STILL NEED TO OPTIMIZE HERE WRT renyiα"""
function Finite_mub(v::T, d::Integer, f::T, N::T, pK::T, n::Integer; analytical_mub::Bool = false, fast::Bool = false) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    # Calculate EC cost per symbol
    leak_EC = EC_cost_mub(v, d, f, N, pK, ϵCR)
    

    """ Here I need an optimization wrt renyiα """ 
    # Optimization wrt Renyi parameter α
    opt_renyi = T(1 +1e-5) # optimal_renyi(v,f,N)
         

    if opt_renyi != 1
        correction = leak_EC + Finite_corrections(renyiα, ϵPE, ϵPA)/N

        # Conic program
        h_renyi = hae_mub_general(v, d, N, pK, n, ϵcompPE, opt_renyi; analytical_mub, fast)

        SKR_Max = h_renyi - correction

        @printf("α-1 = %.5e, SKR = %.2e \n", opt_renyi-T(1), SKR_Max)

    # Otherwise, optimize with respect to renyiα
    else
        finiteSKR_pars = Finite_pars(ϵPA, ϵPE, v, d, N, pK, n, ϵcompPE, leak_EC, analytical_mub, fast)
        optimize_renyi(renyiα) = -FiniteSKR(renyiα[1], finiteSKR_pars)

        # Initial guess
        renyiα0 =[ T(1 +1e-5)]

        α_low = T(1)
        α_high = T(1.1) # A bit tightened, as our numerical analysis indicates
        options = Optim.Options(iterations = 100,f_calls_limit = 30)
        method  = Optim.NelderMead()
        sol = Optim.optimize(optimize_renyi, α_low, α_high, renyiα0, method, options)
        opt_renyi = sol.minimizer[1]
        SKR_Max = -sol.minimum
    end

    return SKR_Max, opt_renyi, leak_EC
end


# d = 5; f = 1.0; N = 1e10; pK = 0.5; n = d + 1;  analytical_mub = true; fast = true; T = Float64; # v = T(0.8);
function Instance_mub(
    d::Integer, 
    f::Real, 
    N::Real, 
    pK::Real; 
    n::Integer = d + 1;
    analytical_mub::Bool = false,
    fast::Bool = false,
    T::DataType = Float64)

    # Enforce desired precision
    f = T(f)
    N = T(N)
    pK = T(pK)

    # Create output file
    RATE_MUB   = "Rate_mub_f"*string(Int(floor(f*100)))*"_d"*string(d)*".csv"
    FILE       = open(RATE_MUB,"a")
    @printf(FILE,"d, f, N, n \n")
    @printf(FILE,"%d, %.2f, %.2f, %d \n",d,f,log10(N),n)
    @printf(FILE,"v, pK, a-1, leakEC, SKR \n")
    close(FILE)

    # Start loop for various values of the visibility
    # Threads.@threads 
    for v ∈ 0.1:0.1:1.0
        @printf("visibility: %.2f ---------\n",v)
        Finite_SKR, opt_renyi, leak_EC = Finite_mub(v, d, f, N, pK, n; analytical_mub, fast)

        # Record outputs
        FILE = open(RATE_MUB,"a")
        @printf(FILE,"%.2f, %.2f, %.6e, %.6f, %.8e \n", v, pK, opt_renyi-T(1), leak_EC, Finite_SKR)
        close(FILE)
    end
end