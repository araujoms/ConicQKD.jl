using LinearAlgebra
using JuMP
using ConicQKD
using Ket
using Optim
import Hypatia
import Hypatia.Cones
import JLD2
using Printf
using Parameters

include("Utils_data_bb84.jl")

@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    ϵCR::T = inv(T(10^11))
    ϵPA::T = 9inv(T(10^11))
    ϵcompPE::T = 9inv(T(10^11))
end

"Alice state after depolarization"
function alice_depol(v::T) where {T<:AbstractFloat}
    dA = 2; dAL = 3
    ω1 = kron(proj(1, dA), proj(1, dAL)) + kron(proj(2, dA), proj(2, dAL))
    ω2 = kron(proj(1, dA), proj(2, dAL)) + kron(proj(2, dA), proj(1, dAL))
    ω01 = kron(ket(1, dA) * ket(2, dA)', ket(1, dAL) * ket(2, dAL)')
    ω10 = kron(ket(2, dA) * ket(1, dA)', ket(2, dAL) * ket(1, dAL)')
    ω = ((1-v/2)*ω1 + v/2*ω2  + (1-v)*(ω01 + ω10))/2
    return ω
end

"Alice state after depolarization and losses"
function alice_depol_loss(v::T, η::T) where {T<:AbstractFloat}
    dA = 2; dAL = 3
    ω = η * alice_depol(v) + (1 - η) * kron(I(dA), proj(3, dAL)) / 2
    return ω
end

"Alice's measurements"
function alice_povm()
    PZ = [proj(1, 2), proj(2, 2)]
    PX = 0.5 * [[1 1; 1 1], [1 -1; -1 1]]
    return vcat(PZ, PX)
end

"Bob's measurements"
function bob_povm(pK::T) where {T<:AbstractFloat}
    QZ = pK * [[1 0 0; 0 0 0; 0 0 0], [0 0 0; 0 1 0; 0 0 0]]
    QX = (1 - pK) * 0.5 * [[1 1 0; 1 1 0; 0 0 0], [1 -1 0; -1 1 0; 0 0 0]]
    Q = [proj(3, 3)]
    return vcat(QZ, QX, Q)
end

"Full Alice's and Bob's POVM"
function POVM_AB(pK::T) where {T<:AbstractFloat}
    A = alice_povm()
    B = bob_povm(pK)
    povm = [kron(a, b) for a ∈ A for b ∈ B]
    return povm
end

"Leakage"
function EC_cost_bb84(qber::T, η::T, f::T, pK::T,ϵCR::T,N::T) where {T<:AbstractFloat}
    # H(A|B) 
    leak_EC = binary_entropy(qber)

    leak_EC *= f * η * pK^2                 # EC efficiency and pK
    leak_EC += ceil(log2(inv(ϵCR)))/N   # Correctness cost added in finite corrections
    return leak_EC
end

"QBER for the Z basis"
function qberZ(v::T, η::T, pK::T) where {T<:AbstractFloat}
    A = alice_povm()
    B = bob_povm(pK)
    ω = alice_depol_loss(v, η)
    p_error = sum(real(dot(kron(A[i], B[j]), ω)) for i ∈ 1:2, j ∈ 1:2 if i != j)
    p_click = sum(real(dot(kron(A[i], B[j]), ω)) for i ∈ 1:2, j ∈ 1:2)
    return p_error / p_click
end

"PE correlations with simulated state"
function PE_probabilities_bb84(v::T, η::T, pK::T) where {T<:AbstractFloat}
    ω = alice_depol_loss(v, η)
    n = size(POVM_AB(pK), 1)
    expval = [real(dot(ω, POVM_AB(pK)[i])) for i ∈ (div(n, 2)+1):n]
    return expval
end

"PE correlations with constraint state"
function constraint_probabilities_bb84(ω::AbstractMatrix, pK::T) where {T<:AbstractFloat}
    n = size(POVM_AB(pK), 1)
    return real(dot.(Ref(ω), POVM_AB(pK)[div(n, 2)+1:n]))
end

function conic_bb84(
    α::T,
    v::T,
    η::T,
    N::T,
    pK::T,
    ϵcompPE::T;
    fast::Bool = true
) where {T<:AbstractFloat}
    dimA = 2
    dimB = 3
    d = dimA * dimB
    n = size(POVM_AB(pK), 1)

    model = GenericModel{T}()

    # Variables
    @variable(model, ωAB[1:d, 1:d], Hermitian)
    @variable(model, qK)
    @variable(model, q[1:div(n, 2)])
    @variable(model, h_QKD)
    @variable(model, h_KL)

    # Constraints on the state
    @constraint(model, partial_trace(ωAB, 2, [2, 3]) == Hermitian(I(2) / 2))

    # Constraints on probabilities
    @constraint(model, sum(q) + qK == 1)

    # Constraints on exp vals via KL divergence
    p_ωAB = constraint_probabilities_bb84(ωAB, pK) * (1 - pK) # PE probabilities

    # Finite bounds via a Bretagnolle-Huber-Carol estimator 
    C_alphbet = 13 # {perp} U {(0,1) x ((X,Z) x (0,1,perp))}
    δ = sqrt((2 * C_alphbet * log(T(2)) - 2 * log(ϵcompPE)) / N)/1e7
    p_sim = PE_probabilities_bb84(v, η, pK) * (1 - pK)
    @constraint(model, [δ; q - p_sim; qK - pK] in Hypatia.EpiNormInfCone{T,T}(1 + 1 + length(q), true))

    # Key map used by both cones
    Ghat_top = [sqrt(pK) * kron(I(2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)')]

    vec_dim = Cones.svec_length(Complex, d)
    ωAB_vec = svec(ωAB)

    # Conic program
    if fast == true
        β = inv(α)
        S = I(6)
        @variable(model, u)

        ZGhat_top = [sqrt(pK) * kron(proj(i, 2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)') for i ∈ 1:2]
        @constraint(model, [u; ωAB_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat_top, ZGhat_top, 1 + vec_dim; S))
        sβ = β < 1 ? -1 : 1
        @constraint(
            model,
            [h_QKD, qK, pK * (sβ * u + 1 - real(tr(Ghat_top[1] * ωAB * Ghat_top[1]')))] in MOI.ExponentialCone()
        )
        @constraint(model, [h_KL; p_ωAB; q] in Hypatia.EpiRelEntropyCone{T}(1 + 2 * length(q), false))
        @objective(model, Min, α * inv(log(T(2)) * (α - 1)) * (h_KL - h_QKD))
    elseif fast == false
        γ = α * inv(2α - 1)
        @variable(model, uTop)
        @variable(model, uBot)
        @variable(model, ψAB[1:d*3, 1:d*3], Hermitian)

        @constraint(model, tr(ψAB) == 1)
        ψ_vec = svec(ψAB)

        # cone for click events
        Zhat_top = [kron(ket(r, 2) * ket(r, 3)', I(6)) for r ∈ 1:2]
        STop = kron(sum(kron(ket(i, 2), proj(i, 2)) for i ∈ 1:2), sum(ket(i, 3) * ket(i, 2)' for i ∈ 1:2))
        @constraint(
            model,
            [uTop; ωAB_vec; ψ_vec] in
            EpiRenyiQKDTriCone{T,Complex{T}}(γ, Ghat_top, Zhat_top, 1 + length(ωAB_vec) + length(ψ_vec); S = STop)
        )

        # cone for no-click events
        Ghat_bottom = [kron(I(2), sqrt(1 - pK) * (proj(1, 3) + proj(2, 3)) + proj(3, 3))]
        Zhat_bottom = [kron(ket(3, 3)', I(6))]
        SBot = I(6)
        @constraint(
            model,
            [uBot; ωAB_vec; ψ_vec] in EpiRenyiQKDTriCone{T,Complex{T}}(
                γ,
                Ghat_bottom,
                Zhat_bottom,
                1 + length(ωAB_vec) + length(ψ_vec);
                S = SBot
            )
        )

        sγ = γ < 1 ? -1 : 1
        @constraint(model, [h_QKD, 1, sγ * (uTop + uBot)] in MOI.ExponentialCone())
        @constraint(model, [h_KL; p_ωAB; pK; q; qK] in Hypatia.EpiRelEntropyCone{T}(1 + 2 + 2 * length(q), false))
        @objective(model, Min, ((α / (α - 1)) * h_KL + (pK - δ) * h_QKD / (γ - 1)) / log(T(2)))
    end

    # Optimize
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Extract results
    h_renyi = objective_value(model)

    return h_renyi
end

"Finite corrections for the final key rate"
Finite_corrections(α::T, ϵPA::T) where {T<:AbstractFloat} =
    (log2(inv(ϵPA))) * α / (α - 1) - 2 

function Finite_bb84(
    dB::Integer,
    N::T,
    v::T,
    pK::T,
    f::T;
    fast::Bool = false
) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵcompPE = epsilon_coeffs{T}()

    η = 10^(-T(dB) / 10)

    # Calculate EC cost per symbol
    qZ = qberZ(v, η, pK)
    leak_EC = EC_cost_bb84(qZ, η, f, pK,ϵCR,N)

    #  Optimization of α parameter 
   obj(αrenyi) = -(conic_bb84(αrenyi, v, η, N, pK, ϵcompPE; fast) - Finite_corrections(αrenyi, ϵPA, ϵCR) / N)

    #ranges
    α_low = T(1 + 1e-6)
    α_high = T(1 + 1e-1)

    println("Starting optimization on α")
    sol = Optim.optimize(obj, α_low, α_high, Brent())

    optimal_renyi = sol.minimizer[1]
    h_renyi = -sol.minimum

    SKR_Max = h_renyi - leak_EC
    @printf("Optimum found for α-1 = %.5e giving a key rate of SKR = %.2e \n", optimal_renyi - 1, SKR_Max)

    return SKR_Max, optimal_renyi, leak_EC, h_renyi
end

"Function to obtain and save the key rates"
function Instance_bb84(f::Real, N::Real, v::Real, ::Type{T}; fast::Bool = true) where {T}

    # Enforce desired precision
    f = T(f)
    N = T(N)

    # Create output file
    RATE_BB84 = "OptRate_bb84_f" * string(f) * "_N1e" * string(count(==('0'), string(Int(N)))) * ".csv"
    file = open(RATE_BB84, "a")
    @printf(file, "f, N, nu \n")
    @printf(file, "%.2f, %.2f, %.2f \n", f, log10(N), v)
    @printf(file, "dB, pK, a-1, leakEC, SKR, dual \n")
    close(file)

    # Start main loop
    for dB ∈ 0:2:48 # Change accordingly
        @printf("Distance: %d ---------\n", L)
        pK = optimal_pK(f,N,L)
        Finite_SKR, optimal_α, leak_EC, dual = Finite_bb84(dB, N, v, pK, f; fast)

        # Record outputs
        file = open(RATE_BB84, "a")
        @printf(file, "%d, %.2f, %.8e, %.12f, %.8e, %.8e \n", dB, pK, optimal_α - T(1), leak_EC, Finite_SKR, dual)
        close(file)
    end
end
