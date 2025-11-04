using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia
import Hypatia.Cones
import JLD2

using Printf
using Parameters
import Optim


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
    d::Integer
    N::T
    pK::T
    m::Integer
    ϵcompPE::T
    leak_EC::T
    fast::Bool
end

function FiniteSKR(α, finiteSKR_pars::Finite_pars{T}) where {T<:AbstractFloat}
    
    # unpack pars
    @unpack ϵPA, ϵPE, v, d, N, pK, m, ϵcompPE, leak_EC, fast = finiteSKR_pars
    
    # Total correction
    correction = leak_EC + Finite_corrections(α, ϵPE, ϵPA)/N

    # Conic program
    h_renyi = hae_mub_general(v, d, N, pK, m, ϵcompPE, α; fast)

    FiniteSecretKey = h_renyi - correction

    # Some log info
    @printf("α-1 = %.5e, SKR = %.2e \n", α-1, FiniteSecretKey)

    return FiniteSecretKey
end

function zgmap(rho::AbstractMatrix, d::Integer)
    K = zgkraus(d)
    zgrho = sum(K[i] * rho * K[i] for i ∈ 1:d)
    return Hermitian(zgrho)
end

function zgkraus(d::Integer)
    K = [kron(proj(i, d), I(d)) for i ∈ 1:d]
    return K
end

Finite_corrections(α::T, ϵPE::T, ϵPA::T) where {T<:AbstractFloat} =
    (log(1/ϵPE)  + log(1/ϵPA))* α/(α-T(1)) - 2

function EC_cost_mub(v::T, d::Integer, f::T, N::T, pK::T, ϵCR::T) where {T<:AbstractFloat}
    
    # H(A|B)
    leak_EC = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(T(d) - 1)
    
    leak_EC *= f*pK^2                 # EC efficiency and pK
    leak_EC += ceil(log2(inv(ϵCR)))/N # Correctness cost

    return leak_EC
end


function simulated_probabilities_mub(v::T, d::Integer, pK::T, m::Integer) where {T<:AbstractFloat}
    
    p2 = ((T(1)-pK)*inv(m-1))^2
    W = v + (1 - v) / d

    # Basis coincidence
    p_sim = p2 * W * ones(m-1)

    # Anything else
    push!(p_sim, T(1) - pK^2 - (m-1)*p2*W)

    return p_sim
end

function constraint_probabilities_mub(ρ::AbstractMatrix, d::Integer, pK::T, m::Integer) where {T<: AbstractFloat}
    mubs = mub(Complex{T}, d) # analytical MUBs from the package Ket

    # Vector of probabilities for each basis
    p   = [pK]
    pPE = [(1-pK)*inv(m-1) for i ∈ 1:m-1]
    append!(p, pPE)

    # Probability of basis coincidence
    p2 = ((1-pK)*inv(m-1))^2
    b = [zeros(Complex{T}, d^2, d^2) for i ∈ 1:m]
    for i ∈ 1:m-1, j ∈ 1:d
        temp = ketbra(mubs[i+1][:, j]) # Note that we skip the first MUB
        b[i] += p2*kron(temp, transpose(temp))
    end

    # Then they sum all other cases
    for i ∈ 1:m, j ∈ 1:m, k ∈ 1:d, l ∈ 1:d
        if i == 1 && j == 1
            continue
        elseif i == j && k == l # This also skips i == 0 (key)
            continue
        end
        tempA = ketbra(mubs[i][:, k])
        tempB = ketbra(mubs[j][:, l])
        b[m] += p[i]*p[j]*kron(tempA, transpose(tempB))
    end

    cleanup!.(b; tol = sqrt(eps(T)))
    b = Hermitian.(b)

    return real(dot.(Ref(ρ), b))
end

function hae_mub_general(
    v::T, 
    d::Integer, 
    N::T, 
    pK::T, 
    m::Integer, 
    ϵcompPE::T, 
    α::T;
    fast::Bool = true
) where {T<:AbstractFloat}
    
    is_complex = true
    model = GenericModel{T}()
    hermitian_space = Ket._sdp_parameters(is_complex)[3]
    R = is_complex ? Complex{T} : T


    # Variables
    @variable(model, ρ[1:d^2, 1:d^2] ∈ hermitian_space)
    @variable(model, q_K)
    @variable(model, q[1:m])
    @variable(model, h_QKD)
    @variable(model, h_KL)


    # Simulated probabilities
    p_sim = simulated_probabilities_mub(v, d, pK, m)
    p_ρAB = constraint_probabilities_mub(ρ, d, pK, m)
    
    # Constraint on states
    @constraint(model, tr(ρ) == 1)

    # Constraints on probabilities
    @constraint(model, sum(q) + q_K == 1)

    # Constraints on exp vals via KL divergence

    # Finite bounds via a Bretagnolle-Huber-Carol estimator
    C_alphbet = length(q)+1 # Key (1) + Coincident bases (m-1) + Non-coincident (1)
    δ = sqrt((2*C_alphbet*log(T(2)) - 2*log(ϵcompPE))/N)
    @constraint(model, [δ; q - p_sim; q_K - pK^2] in Hypatia.EpiNormInfCone{T,T}(1+1+length(q),true))

    # Key map
    Ghat = [I(d^2)]
    Zhat = zgkraus(d)
    blocks = [(i-1)*d+1:i*d for i ∈ 1:d]

    vec_dim = Cones.svec_length(R, d^2)
    ρ_vec = svec(ρ)
    

    # Conic program 
    @variable(model, u)
    if fast
        @constraint(model, [h_KL; p_ρAB; q] in Hypatia.EpiRelEntropyCone{T}(1+2*length(q),false))
        β = inv(α)
        sβ = β < 1 ? -1 : 1
        @constraint(
            model,
            [u; ρ_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat, Zhat, 1 + vec_dim; blocks)
        )
        @constraint(model, [h_QKD, q_K, pK^2 * sβ * u] in MOI.ExponentialCone())
        @objective(model, Min, (α / (α - 1)) * (h_KL - h_QKD) / log(T(2)))
    else
        @constraint(model, [h_KL; p_ρAB; pK^2; q; q_K] in Hypatia.EpiRelEntropyCone{T}(1+2+2*length(q),false))
        γ = inv(2 - inv(α))
        sγ = γ < 1 ? -1 : 1
        dim_σ = size(Zhat[1],2)
        @variable(model, σ[1:dim_σ, 1:dim_σ], Hermitian)
        @constraint(model, tr(σ) == 1)
        σ_vec = svec(σ)
        @constraint(
            model,
            [u; ρ_vec; σ_vec] in EpiRenyiQKDTriCone{T,Complex{T}}(γ, Ghat, Zhat, 1 + 2vec_dim; blocks)
        )
        @constraint(model, [h_QKD, 1, sγ * u] in MOI.ExponentialCone())
        @objective(model, Min, ((α / (α - 1)) * h_KL + (pK^2-δ)*h_QKD / (γ - 1)) / log(T(2)))
    end

    # Optimize
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Extract results
    h_renyi = objective_value(model)

    return h_renyi
end

# Suggested values for a test
# d = 5; f = 1.16; N = 1e9; pK = 0.5; v = 0.9; m = d + 1; fast = true; T = Float64; v = T(0.9);
function Aux_mub(
    d::Integer, 
    f::Real, 
    N::Real, 
    pK::Real,
    v::Real; 
    m::Integer = d + 1,
    fast::Bool = true,
    T::DataType = Float64)

    # Enforce desired precision
    f = T(f)
    N = T(N)
    pK = T(pK)
    v = T(v)

    # Create output file
    RATE_MUB   = "Aux_mub_f"*string(Int(floor(f*100)))*"_d"*string(d)*".csv"
    FILE       = open(RATE_MUB,"a")
    @printf(FILE,"d, f, N, m \n")
    @printf(FILE,"%d, %.2f, %.2f, %d \n",d,f,log10(N),m)
    @printf(FILE,"v, pK, a-1, leakEC, SKR \n")
    close(FILE)

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    # Calculate EC cost per symbol
    leak_EC = EC_cost_mub(v, d, f, N, pK, ϵCR)
    
    b = T(2e-8)
    for i = 1:40
        b += b
        α = T(1) + b
        SKR = FiniteSKR(α, Finite_pars(ϵPA, ϵPE, v, d, N, pK, m, ϵcompPE, leak_EC, fast))
        @printf("α-1 = %.5e, SKR = %.2e \n", b, SKR)


        # Record outputs
        FILE = open(RATE_MUB,"a")
        @printf(FILE,"%.2f, %.2f, %.6e, %.6f, %.8e \n", v, pK, α-T(1), leak_EC, SKR)
        close(FILE)
    end

end
