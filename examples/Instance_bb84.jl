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


@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    ϵCR::T = 1e-11
    ϵPA::T = 9e-11
    ϵPE::T = 9e-11
    ϵcompPE::T = 9e-11
end

@with_kw struct Finite_pars{T<:AbstractFloat}
    L::Integer
    N::T
    Nc::Integer
    pK::T
    renyi::Bool
    fast::Bool
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

"Kraus operator for the pinching map"
function zkraus(dimB::Integer)
    K = [kron(proj(i, 2), I(dimB-1)) for i ∈ 1:2]
    return K
end

"Kraus operator for the key map"
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
function EC_cost_bb84(qber::T, f::T, N::T, pK::T, ϵCR::T) where {T<:AbstractFloat}
    # H(A|B) 
    leak_EC = binary_entropy(qber)

    leak_EC *= N*f*pK^2                 # EC efficiency and pK
    leak_EC += ceil(log2(inv(ϵCR)))/N  # Correctness cost
    return leak_EC
end

"QBER for the Z basis"
function qberZ(v::T, η::T, pK::T) where {T<:AbstractFloat}
    A = alice_povm(pK)
    B = bob_povm(pK)
    ρ = alice_depol_loss(v,η)
    p_error = sum([real(tr(kron(A[i],B[j])*ρ)) for i in 1:2, j in 1:2 if i != j])
    p_click = sum([real(tr(kron(A[i],B[j])*ρ)) for i in 1:2, j in 1:2])
    return p_error/p_click
end

"Finite corrections for the final key rate"
Finite_corrections(α::T, ϵPE::T, ϵPA::T) where {T<:AbstractFloat} =
    (log(1/ϵPE)  + log(1/ϵPA))* α/(α-T(1)) - 2

"Probabilities for key generation"
function probabilities_generation(v::T, η::T, pK::T) where {T<:AbstractFloat} 
    ρ = alice_depol_loss(v,η)
    A = alice_povm(pK)
    B = bob_povm(pK)
    expval = [real(tr(ρ*kron(A[i],B[j]))) for i=1:2, j=1:5]
    return expval
end

function simulated_probabilities_bb84(v::T, η::T, pK::T) where {T<:AbstractFloat} 
    ρ = alice_depol_loss(v,η)
    expval = [real(tr(ρ*ΠAB(pK)[i])) for i=1:size(ΠAB(pK),1)]
    return expval
end

function constraint_probabilities_bb84(ρ::AbstractMatrix, pK::T) where {T<:AbstractFloat}
    return real(dot.(Ref(ρ),ΠAB(pK)))
end


function conic_BB84(
    v      ::T, 
    N      ::T, 
    pK     ::T,
    ϵcompPE::T, 
    α      ::T; 
    renyi  ::Bool = true,
    fast   ::Bool = true
    ) where {T<:AbstractFloat}

    d = dimA*dimB

    model = GenericModel{T}()
    
    # Variables
    @variable(model, ρAB[1:d, 1:d], Hermitian)
    @variable(model, qK ≥ 0)
    @variable(model, q[1:length(ΠAB(pK))] ≥ 0) 
    @variable(model, h_QKD)
    @variable(model, h_KL)

    # Constraints on the marginal state
    ρA = partial_trace(ρAB, 2, [2, 3])
    @constraint(model, ρA == partial_trace(alice_depol_loss(v,η), 2, [2, 3]))

    # Constraints on probabilities
    @constraint(model, sum(q) + qK == 1)

    # Constraints on exp vals via KL divergence
    p_ρAB = constraint_probabilities_bb84(ρAB, pK)
    @constraint(model, [h_KL; p_ρAB[:];pK^2; q[:];qK] in Hypatia.EpiRelEntropyCone{T}(1+2+2*length(q[:]),false))
    
    # Finite bounds via a Bretagnolle-Huber-Carol estimator 
    C_alphbet = 13 # {perp} U {(0,1) x ((X,Z) x (0,1,perp))}
    δ = sqrt((2*C_alphbet*log(2) - 2*log(ϵcompPE))/N)
    p_sim = simulated_probabilities_bb84(v, η, pK)
    @constraint(model, [δ; q[:] - p_sim[:];qK - pK^2] in Hypatia.EpiNormInfCone{T,T}(1+1+length(q[:]),true))

    # Key map
    G = gkraus(pK)
    Ghat =  [I(d)]
    Z = zkraus(dimB)
    Zhat = [Zi*G for Zi in Z]

    blocks = [(i-1)*d+1:i*d for i ∈ 1:d]

    vec_dim = Cones.svec_length(Complex, d)
    ρAB_vec = svec(ρAB)

    # Conic program
    if renyi
        @variable(model, Ψ)
        if fast
            β = inv(α)
            #TODO:  understand and define S
            @constraint(model, [Ψ; ρAB_vec] in EpiFastRenyiQKDTriCone{T,R}(β, Ghat, Zhat, 1 + vec_dim; blocks))
        else
            β = inv(2 - inv(α))
            @variable(model, σAB[1:d, 1:d], Hermitian)
            @constraint(model, tr(σAB) == 1)
            σAB_vec = svec(σAB)
            @constraint(model, [Ψ; ρAB_vec;σAB_vec]) in EpiRenyiQKDTriCone{T,Complex{T}}(β, Ghat,(β, Ghat, Zhat, 1 + 2length(ρ_vec); blocks))
        end
        sβ = β < 1 ? -1 : 1
        @constraint(model, [h_QKD * (β - 1), 1, sβ * u] in MOI.ExponentialCone())
        @objective(model, Min, α*inv(log(T(2))*(α-T(1)))*h_KL + (pK-δ)*inv(log(T(2)))*h_QKD)
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
        h_renyi = dual_objective_value(model)
    else
        throw("Not implemented yet")
    end
    return h_renyi 
end


function FiniteSKR(α, finiteSKR_pars::Finite_pars{T}) where {T<:AbstractFloat}
    
    # unpack pars
    @unpack L, N, Nc, pK, renyi, fast = finiteSKR_pars
    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    # Calculate EC cost per symbol
    qZ = qberZ(v, η, pK)
    leak_EC = EC_cost_bb84(qZ, f, N, pK, ϵCR)
    
    # Total correction
    correction = leak_EC + Finite_corrections(α, ϵPE, ϵPA)/N

    # Conic program
    h_renyi = conic_bb84(v,α,N, pK, ϵcompPE )

    FiniteSecretKey = h_renyi - correction

    # Some log info
    @printf("α-1 = %.5e, SKR = %.2e \n", renyiα-1, FiniteSecretKey)

    return FiniteSecretKey
end

function Finite_bb84(L::Integer, N::T, pK::T, Nc::Integer; renyi::Bool = false) where {T<:AbstractFloat}

    # Optimization wrt Renyi parameter α
    finiteSKR_pars = Finite_pars(L, N, Nc, pK, renyi, fast)
    optimize_renyi(α) = -FiniteSKR(α[1], finiteSKR_pars)

    α0 = [ T(1 +1e-6)]
    α_low = T(1); α_high = T(1.1) # A bit tightened, as our numerical analysis indicates
    options = Optim.Options(iterations = 100,f_calls_limit = 30)
    method  = Optim.NelderMead()
    sol = Optim.optimize(optimize_renyi, α_low, α_high, α0 ,method,options)
    optimal_renyi = sol.minimizer[1]
    SKR_Max = -sol.minimum

    return SKR_Max, optimal_renyi, leak_EC
end

function Instance_bb84(
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
    RATE_BB84 = "Rate_bb84_Nc"*string(Nc)*".csv"
    file      = open(RATE_BB84,"a")
    @printf(file,"xi, f, Nc, N, nu, eta \n")
    @printf(file,"0.01, %.2f, %d, %.2f, %.2f, %.2f \n",f,Nc,log10(N))
    @printf(file,"D, pK, a-1, leakEC, SKR \n")
    close(file)

    # Start loop for various values of the distance
    # Threads.@threads 
    for L ∈ vcat(1,5:5:40)
        @printf("Distance: %d ---------\n",L)
        Finite_SKR, optimal_α, leak_EC  = Finite_dmcv(L, f, N, pK, Nc; renyi = true)

        # Record outputs
        file = open(RATE_BB84,"a")
        @printf(file,"%d, %.2f, %.2f, %.8e, %.12f, %.8e \n",L,pK,optimal_α-T(1),leak_EC,Finite_SKR)
        close(file)
    end
end


f = 1.16; N = 1e10; pK = 0.73; Nc = 5; T = Float64; L= 20; renyi = true;fast=true
v=0.03;η=0.8; 
dimA = 2
dimB = 3


# Instance_bb84(f,N,pK,Nc;T)