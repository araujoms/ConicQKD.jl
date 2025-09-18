using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia
import Hypatia.Cones
import JLD2

using Printf
using Parameters

dimA = 2
dimB = 3


@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    ϵCR::T = 1e-11
    ϵPA::T = 9e-11
    ϵPE::T = 9e-11
    ϵcompPE::T = 9e-11
end


"Alice state after depolarization"
function alice_depol(v,dAL)
    dA=2; 
    ρ1 = kron(proj(1,dA),proj(1,dAL)) + kron(proj(2,dA),proj(2,dAL))
    ρ2 = kron(proj(1,dA),proj(2,dAL)) + kron(proj(2,dA),proj(1,dAL))
    ρ01 = kron(ket(1,dA)*ket(2,dA)',ket(1,dAL)*ket(2,dAL)') 
    ρ10 = kron(ket(2,dA)*ket(1,dA)',ket(2,dAL)*ket(1,dAL)')
    ρ = ((1-v/2)*ρ1 + v/2*ρ2  + (1-v)*(ρ01 + ρ10))/2
    return ρ
end

"Alice state after depolarization and losses"
function alice_depol_loss(v::T, η::T) where {T<:AbstractFloat}
    dA=2; dAL= 3
    ρ = η*alice_depol(v,3)+ (1-η)*kron(I(dA),proj(3,dAL))/2
    return ρ
end

# ----------  Choi representation of losses ---------- #
"Choi operator for channel losses"
function choi_loss(η)
    dA=2; dAL= 3
    ϕplus= kron(ket(1,dA),ket(1,dAL)) + kron(ket(2,dA),ket(2,dAL)) 
    L = kron(proj(1,dA),proj(3,dAL)) + kron(proj(2,dA),proj(3,dAL))
    Φ = η*ϕplus*ϕplus' + (1-η)*L
    return Φ
end

function state(η,v)
    ρaA = partial_transpose(alice_depol(v,2),[2],[2,2])
    J = choi_loss(η)
    ρout = real(partial_trace(kron(ρaA,I(3))*kron(I(2),J), 2,[2,2,3]))
    return ρout
end
# ----------------------------------------------------- #

function zkraus(dimB::Integer)
    K = [kron(proj(i, 2), I(dimB-1)) for i ∈ 1:2]
    return K
end

function gkraus(pK::T) where {T<:AbstractFloat}
    G = sqrt(pK)*kron(I(2), [1 0 0; 0 1 0])
    return G
end

"Alice's measurements"
function alice_povm(pK::T) where {T<:AbstractFloat}
    PZ = pK*[proj(1,2), proj(2,2)]
    PX = (1-pK)*0.5*[[1 1; 1 1], [1 -1; -1 1]]
    return vcat(PZ, PX)
end

"Bob's measurements"
function bob_povm(pK::T) where {T<:AbstractFloat}
    QX =(1-pK)/2 .*[[1 1 0; 1 1 0; 0 0 0],[1 -1 0; -1 1 0; 0 0 0]]
    QZ = pK.*[[1 0 0;0 0 0;0 0 0],[0 0 0; 0 1 0; 0 0 0]]
    Q = [proj(3,3)]
    return vcat(QZ,QX,Q)
end

"Full Alice's and Bob's POVM"
function ΠAB(pK::T) where {T<:AbstractFloat}
    A = alice_povm(pK)
    B = bob_povm(pK)
    povm = [kron(a,b) for a in A for b in B]
    return povm
end

"Leackage"
function EC_cost_bb84(a::Integer, f::T, N::T, pK::T, ϵCR::T) where {T<:AbstractFloat}
    # H(A|B) 
    leak_EC = 1-binary_entropy(a)

    leak_EC *= N*f*pK^2                 # EC efficiency and pK
    leak_EC += ceil(log2(inv(ϵCR)))/N  # Correctness cost
    return leak
end

"Finite corrections for the final key rate"
Finite_corrections(α::T, ϵPE::T, ϵPA::T) where {T<:AbstractFloat} =
    (log(1/ϵPE)  + log(1/ϵPA))* α/(α-T(1)) - 2


function simulated_probabilities_bb84(v::T, η::T, pK::T) where {T<:AbstractFloat} 
    ρ = alice_depol_loss(v,η)
    expval = [real(tr(ρ*ΠAB(pK)[i])) for i=1:size(ΠAB(pK),1)]
    return expval
end

function constraint_probabilities_bb84(ρ::AbstractMatrix, pK::T) where {T<:AbstractFloat}
    return real(dot.(Ref(ρ),ΠAB(pK)))
end



function conic_BB84(v::T, N::T, pK::T, n::Integer, ϵcompPE::T, α::T; renyi::Bool = false) where {T<:AbstractFloat}
    d = dimA*dimB
    model = GenericModel{T}()
    @variable(model, ρ[1:d, 1:d], Hermitian)
    R = Complex{T}

    # Optimized probabilities
    @variable(model, qK ≥ 0)
    @variable(model, q[1:length(ΠAB(pK))] ≥ 0) 
    @constraint(model, sum(q) + qK == 1)

    # Simulated probabilities 
    p_sim = simulated_probabilities_bb84(v, η, pK)
    p_ρAB = constraint_probabilities_bb84(ρ, pK)

    # Add constraints
    @constraint(model, tr(ρ)==T(1))

    # Finite bounds via a Bretagnolle-Huber-Carol estimator 
    C_alphbet = 13 # TODO: check
    δ = sqrt((2*C_alphbet*log(2) - 2*log(ϵcompPE))/N)
    @constraint(model, [δ; q[:] - p_sim[:];qK - pK^2] in Hypatia.EpiNormInfCone{T,T}(1+1+length(q[:]),true))


    G = gkraus(pK)
    Ghat = I(d) # TODO: check
    Z = zkraus(dimB)
    Zhat = [Zi*G for Zi in Z]
    blocks = [(i-1)*d+1:i*d for i ∈ 1:d] # TODO: checkear esto

    vec_dim = Cones.svec_length(R, d)
    ρ_vec = svec(ρ)
    

    # KL cone #TODO revisar eso, pero parece que está bien...
    @variable(model, h_KL)
    @constraint(model, [h_KL; p_ρAB[:];pK^2; q[:];qK] in Hypatia.EpiRelEntropyCone{T}(1+2+2*length(q[:]),false))


    # QKD (Rényi) cone 
    @variable(model, Ψ)
    if renyi
        @variable(model, h)
        β = inv(2 - inv(α))
        sβ = β < 1 ? -1 : 1
        @constraint(model, [Ψ; ρ_vec] in EpiRenyiQKDTriCone{T,R}(β, Ghat, Zhat, 1 + vec_dim; blocks))
        @constraint(model, [h * (β - 1), 1, sβ * Ψ] in MOI.ExponentialCone())
        # TODO en paper aparece qK - δ!!! 
        @objective(model, Min, α*inv(log(T(2))*(α-T(1)))*h_KL + (pK^2-δ)*inv(log(T(2)))*h)
    else
        throw("Not implemented yet")
        # @constraint(model, [Ψ; ρ_vec] in EpiQKDTriCone{T,R}(Ghat, Zhatperm, 1 + vec_dim; blocks))
    end

    # Optimize
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Extract results
    if renyi
        ObjVal = dual_objective_value(model)
    else
        throw("Not implemented yet")
    end
    return ObjVal
end

function Finite_bb84(L::Integer, f::T, N::T, pK::T, Nc::Integer, Δs::T, Δ::T; renyi::Bool = false) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    # Pick amplitude for the coherent states
    γ = L == 20 ? 0.77 : 0.8

    # Calculate EC cost per symbol
    leak_EC = EC_cost_bb84(a, f, N, pK, ϵCR)

    """ Here I need an optimization wrt α """ 
    α = L == 20 ? T(1 +1.911e-5) : T(1 +5e-4) # Test value
    # Total correction
    correction = leak_EC + Finite_corrections(α, ϵPE, ϵPA)/N

    # Conic program
    h_renyi = conic_bb84(L, N, pK, Nc, ϵcompPE, γ, α ; renyi)

    Finite_SKR = h_renyi - correction
    return Finite_SKR, α, γ, leak_EC
end

# f = 1.0; N = 1e10; pK = 0.73; Nc = 5; T = Float64; L= 20; renyi = true;
# v=0.03;η=0.8; 
function Instance_dmcv(
    f::Real,
    N::Real,
    pK::Real,
    Nc::Integer;
    T::DataType=Float64
    )

    ### Friendly reminder of basic parameters
    # ξ     = 0.01 (excess noise in SNUs)
    # α_att = 0.2  (attenuation at the fiber in dB/km)

    # Enforce desired precision
    f = T(f)
    N = T(N)
    pK = T(pK)

    # Create output file
    RATE_BB84   = "Rate_bb84_f"*string(Int(floor(f*100)))*".csv"
    FILE       = open(RATE_BB84,"a")
    @printf(FILE_RATE,"xi, f, Nc, N, nu, eta \n")
    @printf(FILE_RATE,"0.01, %.2f, %d, %.2f, %.2f, %.2f \n",f,Nc,log10(N))
    @printf(FILE_RATE,"D, amp, pK, a-1, leakEC, SKR \n")
    close(FILE)

    # Start loop for various values of the distance
    # Threads.@threads 
    for L ∈ [1,5:5:40]
        @printf("Distance: %d ---------\n",L)
        Finite_SKR, optimal_α, γ, leak_EC  = Finite_dmcv(L, f, N, pK, Nc; renyi = true)

        # Record outputs
        FILE = open(RATE_BB84,"a")
        @printf(FILE_RATE,"%d, %.2f, %.2f, %.8e, %.12f, %.8e \n",L,γ,pK,optimal_α-T(1),leak_EC,Finite_SKR)
        close(FILE)
    end
end