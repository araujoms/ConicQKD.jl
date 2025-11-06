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
    ϵPE::T = 9inv(T(10^11))
    ϵcompPE::T = 9inv(T(10^11))
end

"Alice state after depolarization"
function alice_depol(v, dAL)
    dA = 2
    ρ1 = kron(proj(1, dA), proj(1, dAL)) + kron(proj(2, dA), proj(2, dAL))
    ρ2 = kron(proj(1, dA), proj(2, dAL)) + kron(proj(2, dA), proj(1, dAL))
    ρ01 = kron(ket(1, dA) * ket(2, dA)', ket(1, dAL) * ket(2, dAL)')
    ρ10 = kron(ket(2, dA) * ket(1, dA)', ket(2, dAL) * ket(1, dAL)')
    ρ = ((1 + v) / 2 * ρ1 + (1 - v) / 2 * ρ2 + v * (ρ01 + ρ10)) / 2
    return ρ
end

"Alice state after depolarization and losses"
function alice_depol_loss(v::T, η::T) where {T<:AbstractFloat}
    dA = 2
    dAL = 3
    ρ = η * alice_depol(v, 3) + (1 - η) * kron(I(dA), proj(3, dAL)) / 2
    return ρ
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

"Kraus operator for the pinching map after facial reduction"
function zkraus()
    QB_Z = (proj(1, 3) + proj(2, 3))
    QB_perp = proj(3, 3)
    Z = [kron(proj(1, 2), QB_Z) + kron(I(2), QB_perp), kron(proj(2, 2), QB_Z)]
    return Z
end

"Kraus operator for the key map"
function gkrausTop(pK::T) where {T<:AbstractFloat}
    QB_Z = proj(1, 3) + proj(2, 3)
    G = sqrt(pK) * sum(kron(ket(i, 3), kron(proj(i, 2), QB_Z)) for i ∈ 1:2)
    return G
end

"Leakage"
function EC_cost_bb84(qber::T, η::T, f::T, pK::T) where {T<:AbstractFloat}
    # H(A|B) 
    leak_EC = binary_entropy(qber)

    leak_EC *= f * η * pK^2                 # EC efficiency and pK
    # leak_EC += ceil(log2(inv(ϵCR)))/N   # Correctness cost added in finite corrections
    return leak_EC
end

"QBER for the Z basis"
function qberZ(v::T, η::T, pK::T) where {T<:AbstractFloat}
    A = alice_povm()
    B = bob_povm(pK)
    ρ = alice_depol_loss(v, η)
    p_error = sum(real(dot(kron(A[i], B[j]), ρ)) for i ∈ 1:2, j ∈ 1:2 if i != j)
    p_click = sum(real(dot(kron(A[i], B[j]), ρ)) for i ∈ 1:2, j ∈ 1:2)
    return p_error / p_click
end

"PE correlations with simulated state"
function PE_probabilities_bb84(v::T, η::T, pK::T) where {T<:AbstractFloat}
    ρ = alice_depol_loss(v, η)
    n = size(POVM_AB(pK), 1)
    expval = [real(dot(ρ, POVM_AB(pK)[i])) for i ∈ (div(n, 2)+1):n]
    return expval
end

"PE correlations with constraint state"
function constraint_probabilities_bb84(ρ::AbstractMatrix, pK::T) where {T<:AbstractFloat}
    n = size(POVM_AB(pK), 1)
    return real(dot.(Ref(ρ), POVM_AB(pK)[div(n, 2)+1:n]))
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
    @variable(model, ρAB[1:d, 1:d], Hermitian)
    @variable(model, qK)
    @variable(model, q[1:div(n, 2)])
    @variable(model, h_QKD)
    @variable(model, h_KL)

    # Constraints on the state
    @constraint(model, partial_trace(ρAB, 2, [2, 3]) == Hermitian(I(2) / 2))

    # Constraints on probabilities
    @constraint(model, sum(q) + qK == 1)

    # Constraints on exp vals via KL divergence
    p_ρAB = constraint_probabilities_bb84(ρAB, pK) * (1 - pK) # PE probabilities

    # Finite bounds via a Bretagnolle-Huber-Carol estimator 
    C_alphbet = 13 # {perp} U {(0,1) x ((X,Z) x (0,1,perp))}
    δ = sqrt((2 * C_alphbet * log(T(2)) - 2 * log(ϵcompPE)) / N)
    p_sim = PE_probabilities_bb84(v, η, pK) * (1 - pK)
    @constraint(model, [δ; q - p_sim; qK - pK] in Hypatia.EpiNormInfCone{T,T}(1 + 1 + length(q), true))

    # Key map
    G_top = gkrausTop(pK)
    Ghat_top = [sqrt(pK) * kron(I(2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)')]

    vec_dim = Cones.svec_length(Complex, d)
    ρAB_vec = svec(ρAB)

    # Conic program
    if fast == true
        @constraint(model, [h_KL; p_ρAB; q] in Hypatia.EpiRelEntropyCone{T}(1 + 2 * length(q), false))
        β = inv(α)
        S = I(6)
        @variable(model, u)

        ZGhat_top = [sqrt(pK) * kron(proj(i, 2), ket(1, 2) * ket(1, 3)' + ket(2, 2) * ket(2, 3)') for i ∈ 1:2]
        @constraint(model, [u; ρAB_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat_top, ZGhat_top, 1 + vec_dim; S))
        sβ = β < 1 ? -1 : 1
        @constraint(
            model,
            [h_QKD, qK, pK * (sβ * u + 1 - real(tr(Ghat_top[1] * ρAB * Ghat_top[1]')))] in MOI.ExponentialCone()
        )
        @objective(model, Min, α * inv(log(T(2)) * (α - 1)) * (h_KL - h_QKD))
    elseif fast == false
        @constraint(model, [h_KL; p_ρAB; pK; q; qK] in Hypatia.EpiRelEntropyCone{T}(1 + 2 + 2 * length(q), false))
        γ = α * inv(2α - 1)
        #variables
        @variable(model, uTop)
        @variable(model, uBot)
        @variable(model, ψAB[1:d*3, 1:d*3], Hermitian)

        #constraint
        @constraint(model, tr(ψAB) == 1)

        ψ_vec = svec(ψAB)

        # cone for click events
        Zhat_top = [kron(ket(r, 2) * ket(r, 3)', I(6)) for r ∈ 1:2]
        STop = kron(sum(kron(ket(i, 2), proj(i, 2)) for i ∈ 1:2), sum(ket(i, 3) * ket(i, 2)' for i ∈ 1:2))
        @constraint(
            model,
            [uTop; ρAB_vec; ψ_vec] in
            EpiRenyiQKDTriCone{T,Complex{T}}(γ, Ghat_top, Zhat_top, 1 + length(ρAB_vec) + length(ψ_vec); S = STop)
        )

        # cone for no-click events
        Ghat_bottom = [kron(I(2), sqrt(1 - pK) * (proj(1, 3) + proj(2, 3)) + proj(3, 3))]
        Zhat_bottom = [kron(ket(3, 3)', I(6))]
        SBot = I(6)
        @constraint(
            model,
            [uBot; ρAB_vec; ψ_vec] in EpiRenyiQKDTriCone{T,Complex{T}}(
                γ,
                Ghat_bottom,
                Zhat_bottom,
                1 + length(ρAB_vec) + length(ψ_vec);
                S = SBot
            )
        )

        sγ = γ < 1 ? -1 : 1
        @constraint(model, [h_QKD, 1, sγ * (uTop + uBot)] in MOI.ExponentialCone())
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
Finite_corrections(α::T, ϵPE::T, ϵPA::T, ϵCR::T) where {T<:AbstractFloat} =
    (log2(inv(ϵPA))) * α / (α - 1) - 2 + ceil(log2(inv(ϵCR)))

# (log2(inv(1e-80/2)) )* α/(α-T(1)) - 2 + ceil( log2(inv(1e-80/2)) )

function Finite_bb84(
    dB::Integer,
    N::T,
    v::T,
    pK::T,
    f::T;
    fast::Bool = false
) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    η = 10^(-T(dB) / 10)

    # Calculate EC cost per symbol
    qZ = qberZ(v, η, pK)
    leak_EC = EC_cost_bb84(qZ, η, f, pK)

    # #---------------- Optimization of α paramter ---------------------------
    obj(αrenyi) = -conic_bb84(αrenyi, v, η, N, pK, ϵcompPE; fast)

    #ranges
    α_low = T(1 + 1e-8)
    α_high = T(1.01)
    α0 = T(1 + 2e-5)

    println("Starting optimization on α")
    sol = Optim.optimize(obj, α_low, α_high, Brent())

    optimal_renyi = sol.minimizer[1]
    h_renyi = -sol.minimum

    # ------------------------------------------
    # # FINE SEARCH
    # α_min,α_max  = 1e-7, 0.01
    # n = 300 
    # α_grid =1 .+ α_min .* ((α_max/α_min) .^ (range(0, 1; length=n)))

    # println("Starting loop on α")

    # SKR_vals = [conic_bb84(α,v,η,N, pK, ϵcompPE;renyi, fast) for α in α_grid]
    # SKR_vals = filter(!isnan, SKR_vals)

    # h_renyi, idx = findmax(SKR_vals)
    # optimal_renyi = α_grid[idx]

    #------------------------------------------

    correction = leak_EC + Finite_corrections(optimal_renyi, ϵPE, ϵPA, ϵCR) / N
    SKR_Max = h_renyi - correction
    @printf("Optimum found for α-1 = %.5e giving a key rate of SKR = %.2e \n", optimal_renyi - 1, SKR_Max)

    return SKR_Max, optimal_renyi, leak_EC, h_renyi
end

"Function to obtain and save the key rates"
function Instance_bb84(f::Real, N::Real, v::Real, ::Type{T}) where {T}

    # Enforce desired precision
    f = T(f)
    N = T(N)

    # Create output file
    RATE_BB84 = "OptRate_bb84_f" * string(f) * "_N1e" * string(count(==('0'), string(Int(N)))) * ".csv"
    file = open(RATE_BB84, "a")
    @printf(file, "f, N, nu \n")
    @printf(file, "%.2f, %.2f, %.2f \n", f, log10(N), v)
    @printf(file, "D, pK, a-1, leakEC, SKR, dual \n")
    close(file)

    # Start loop for various values of the distance
    for L ∈ 0:2:36 # ∈ vcat(1,10:10:40)
        @printf("Distance: %d ---------\n", L)
        pK = 0.9 #optimal_pK(f,N,L)
        Finite_SKR, optimal_α, leak_EC, dual = Finite_bb84(L, N, v, pK, f; renyi = true, fast = true)

        # Record outputs
        file = open(RATE_BB84, "a")
        @printf(file, "%d, %.2f, %.8e, %.12f, %.8e, %.8e \n", L, pK, optimal_α - T(1), leak_EC, Finite_SKR, dual)
        close(file)
    end
end

############### REMOVE FUNCTIONS BELOW ################################
"Auxiliary function to optimize on pK (to be remove)"
function Instance_bb84_pK(dB::Real, f::Real, N::Real, v::Real, ::Type{T}; fast::Bool = true) where {T}

    # Enforce desired precision
    f = T(f)
    N = T(N)

    # Create output file
    if fast == true
        RATE_BB84 =
            "examples/data_bb84/varying_pK/delta0_f" *
            string(f) *
            "_Optim_bb84_N1e" *
            string(count(==('0'), string(Int(N)))) *
            "_dB" *
            string(dB) *
            ".csv"
    else
        RATE_BB84 =
            "examples/data_bb84/true_cone/f" *
            string(f) *
            "_Optim_bb84_N1e" *
            string(count(==('0'), string(Int(N)))) *
            "_dB" *
            string(dB) *
            ".csv"
    end
    file = open(RATE_BB84, "a")
    @printf(file, "f, N, nu \n")
    @printf(file, "%.2f, %.2f, %.2f \n", f, log10(N), v)
    @printf(file, "D, pK, a-1, leakEC, SKR, dual \n")
    close(file)

    if dB < 10
        range = 0.85:0.01:0.98
    else
        range = 0.06:0.01:0.8
    end
    # Start loop for various values of the distance
    for pK ∈ 0.6:0.01:0.98# ∈ vcat(1,10:10:40)
        @printf("Values of pK: %d ---------\n", dB)
        Finite_SKR, optimal_α, leak_EC, dual = Finite_bb84(dB, N, v, pK, f; renyi = true, fast)

        # Record outputs
        file = open(RATE_BB84, "a")
        @printf(file, "%d, %.2f, %.8e, %.12f, %.8e, %.8e \n", dB, pK, optimal_α - T(1), leak_EC, Finite_SKR, dual)
        close(file)
    end
end

function Finite_bb84_alpha(
    α::T,
    dB::Integer,
    N::T,
    v::T,
    pK::T,
    f::T;
    fast::Bool = false
) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    η = 10^(-T(dB) / 10)

    # Calculate EC cost per symbol
    qZ = qberZ(v, η, pK)
    leak_EC = EC_cost_bb84(qZ, η, f, pK)
    h_renyi = conic_bb84(α, v, η, N, pK, ϵcompPE; fast)
    correction = leak_EC + Finite_corrections(α, ϵPE, ϵPA, ϵCR) / N
    SKR_Max = h_renyi - correction
    @printf("Optimum found for α-1 = %.5e giving a key rate of SKR = %.2e \n", α - 1, SKR_Max)

    return SKR_Max, leak_EC, h_renyi
end

"Auxiliary function to optimize on α (to be remove)"
function Instance_bb84_alpha(dB::Real, f::Real, N::Real, v::Real, ::Type{T}; fast::Bool = true) where {T}

    # Enforce desired precision
    f = T(f)
    N = T(N)
    pK = 0.9

    # Create output file
    if fast == true
        RATE_BB84 =
            "examples/data_bb84/varying_alpha/f" *
            string(f) *
            "_Optim_bb84_N1e" *
            string(count(==('0'), string(Int(N)))) *
            "_dB" *
            string(dB) *
            ".csv"
    else
        RATE_BB84 =
            "examples/data_bb84/varying_alpha/Renyi_f" *
            string(f) *
            "_Optim_bb84_N1e" *
            string(count(==('0'), string(Int(N)))) *
            "_dB" *
            string(dB) *
            ".csv"
    end
    file = open(RATE_BB84, "a")
    @printf(file, "f, N, nu \n")
    @printf(file, "%.2f, %.2f, %.2f \n", f, log10(N), v)
    @printf(file, "D, pK, a-1, leakEC, SKR, dual \n")
    close(file)

    # Start loop for various values of the distance

    b = 2inv(T(10^8))
    for i ∈ 1:40
        b += b
        α = 1 + b
        @printf("Values of α: %.8e ---------\n", α)
        Finite_SKR, leak_EC, dual = Finite_bb84_alpha(α, dB, N, v, pK, f; fast)

        # Record outputs
        file = open(RATE_BB84, "a")
        @printf(file, "%d, %.2f, %.8e, %.12f, %.8e, %.8e \n", dB, pK, α - 1, leak_EC, Finite_SKR, dual)
        close(file)
    end
end

#f=1.16; N = 1e9;  T = Float64;# pK=0.90;# pK = 0.95;#L= 20;
#v=0.97;

# Instance_bb84(f,N,v;T)

# Instance_bb84_alpha(0,f,N,v;T, fast=false)

#for dB=[0,8,16] #
#    Instance_bb84_pK(dB,f,N,v;T, fast=false)
#end

# L= 0; pK=0.91; η=1.;  α=1.00004000 5.84726152e-05+1
