using ConicQKD
using SpecialFunctions
using LinearAlgebra
using JuMP
using Ket
import Hypatia
import Hypatia.Cones
import Integrals

using Printf
using Parameters
import Optim

include("Utils_data_dmcv.jl")

@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    ϵCR::T = 1e-11
    ϵPA::T = 9e-11
    ϵPE::T = 9e-11
    ϵcompPE::T = 9e-11
end

@with_kw struct Finite_pars{T<:AbstractFloat}
    ϵPA::T
    ϵPE::T
    L::Integer
    N::T
    Nc::Integer
    pK::T
    Δs::T
    Δ::T
    ϵcompPE::T
    amplitude::T
    leak_EC::T
    fast::Bool
end

function FiniteSKR(α, finiteSKR_pars::Finite_pars{T}) where {T<:AbstractFloat}
    
    # unpack pars
    @unpack ϵPA, ϵPE, L, N, Nc, pK, Δs, Δ, ϵcompPE, amplitude, leak_EC, fast = finiteSKR_pars
    
    # Total correction
    correction = leak_EC + Finite_corrections(α, ϵPE, ϵPA)/N

    # Conic program
    h_renyi = hbe_dmcv_general(L, N, Nc, pK, Δs, Δ, ϵcompPE, amplitude, α; fast)

    FiniteSecretKey = h_renyi - correction

    # Some log info
    @printf("α-1 = %.5e, SKR = %.2e \n", α-1, FiniteSecretKey)

    return FiniteSecretKey
end


function alice_part(amplitude::Real)
    ρ = Hermitian(ones(Complex{typeof(amplitude)},4,4))
    ρ.data[1,2] = (exp(-(1+im)*amplitude^2))
    ρ.data[1,3] = (exp(-2*amplitude^2))
    ρ.data[1,4] = (exp(-(1-im)*amplitude^2))
    ρ.data[2,3] = ρ.data[1,2]
    ρ.data[2,4] = ρ.data[1,3]
    ρ.data[3,4] = ρ.data[1,2]
    ρ *= 0.25
end

function integrand(vars, pars)
    ζ, θ = vars
    ξ, η, x, amplitude = pars
    return ζ * exp(-abs2(ζ * exp(im * θ) - sqrt(η) * im^x * amplitude) / (1 + η * ξ / 2))
end

function integrate(bounds, pars)
    T = eltype(pars)
    problem = Integrals.IntegralProblem(integrand, bounds, pars)
    tol = T == Float64 ? eps(T)^(3 / 4) : sqrt(eps(T))
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = tol, abstol = tol)
    return sol.u
end

function joint_probability(L::Integer, ξ::T, amplitude::T) where {T<:AbstractFloat}
    α_att = T(2)/10
    η = 10^(- α_att*L / 10)
    pAB = zeros(T, 4, 4)
    for x ∈ 0:3
        pars = [ξ, η, x, amplitude]
        for z ∈ 0:3
            bounds = ([T(0), T(π) * (2 * z - 1) / 4], [T(Inf), T(π) * (2 * z + 1) / 4])
            pAB[x+1, z+1] = integrate(bounds, pars)
        end
    end
    pAB ./= 4 * T(π) * (1 + η * ξ / 2)
    return pAB
end

function hba_dmcv(L::Integer, ξ::T, amplitude::T) where {T<:AbstractFloat}
    pAB = joint_probability(L, ξ, amplitude)
    pBA = transpose(pAB)
    return conditional_entropy(pBA)
end


function alice_part(α::T) where {T<:Real}
    ρ = zeros(Complex{T}, 4, 4)
    for j ∈ 0:3, i ∈ 0:j
        ρ[i+1, j+1] = 0.25 * exp(-α^2 * (1 - (1.0 * im)^(i - j)))
    end
    return Hermitian(ρ)
end

function sinkpi4(::Type{T}, k::Integer) where {T<:Real} #computes sin(k*π/4) with high precision
    if mod(k, 4) == 0
        return T(0)
    else
        signal = T((-1)^div(k, 4, RoundDown))
        if mod(k, 2) == 0
            return signal
        else
            return signal / sqrt(T(2))
        end
    end
end


function test_basis_dmcv(Nc::Integer, Δs::T, Δ::T) where {T<:AbstractFloat}
    R = [Hermitian(zeros(Complex{T},Nc+1,Nc+1)) for z=0:5]
    for z = 0:3
        for n=0:Nc
            for m=n:Nc
                if n == m
                    R[z+1][n+1,m+1] = (gamma(T(1+n)) - gamma(T(1+n),Δs^2))/(4*gamma(T(1+n)))
                else
                    angular = 2*im^(mod(z*(n-m),4))*sinkpi4(T,n-m)/(n-m)
                    radial = (gamma(1 + T(n+m)/2) - gamma(1 + T(n+m)/2,Δs^2))/(2*T(π)*sqrt(gamma(T(1+n))*gamma(T(1+m))))                    
                    R[z+1].data[n+1,m+1] = angular*radial
                end
            end
        end
    end
    for n=0:Nc
        R[5][n+1,n+1] = (gamma(T(1+n),Δs^2) - gamma(T(1+n),Δ^2))/gamma(T(1+n))
        R[6][n+1,n+1] = gamma(T(1+n),Δ^2)/gamma(T(1+n))
    end
    return R
end


function key_basis_dmcv(::Type{T}, Nc::Integer) where {T<:AbstractFloat}
    R = [Hermitian(zeros(Complex{T},Nc+1,Nc+1)) for z=0:3]
    for z = 0:3
        for n=0:Nc
            for m=n:Nc
                if n == m
                    R[z+1][n+1,m+1] = T(1)/4
                else
                    angular = 2*im^(mod(z*(n-m),4))*sinkpi4(T,n-m)/(n-m)
                    radial = gamma(1 + T(n+m)/2)/(2*T(π)*sqrt(gamma(T(1+n))*gamma(T(1+m))))                    
                    R[z+1].data[n+1,m+1] = angular*radial
                end
            end
        end
    end
    return R
end


function gkraus(::Type{T}, Nc::Integer) where {T<:Real}
    sqrtbasis = sqrt.(key_basis_dmcv(T, Nc))
    V = sum(kron(I(4), sqrtbasis[i], ket(i, 4)) for i ∈ 1:4)
    return V
end

function zkraus(Nc::Integer)
    K = [kron(I(4 * (Nc + 1)), proj(i, 4)) for i ∈ 1:4]
    return K
end


function simulated_probabilities_dmcv(Δs::T, Δ::T, amplitude::T, L::Integer) where {T<:AbstractFloat}
    α_att = T(2)/10
    ξ = T(1)/100
    η = 10^(-(α_att*L)/10)
    p_sim = zeros(T,4,6)
    for x=0:3
        pars = [ξ, η, x, amplitude]
        for z = 0:3
            bounds = ([T(0), T(π)*(2*z-1)/4], [T(Δs), T(π)*(2*z+1)/4])
            p_sim[x+1,z+1] = integrate(bounds,pars)
        end
        #z = 4
        bounds = ([T(Δs), T(0)], [T(Δ), 2*T(π)])
        p_sim[x+1,5] = integrate(bounds,pars)
        #z = 5
        bounds = ([T(Δ), T(0)], [T(Inf), 2*T(π)])
        p_sim[x+1,6] = integrate(bounds,pars)
    end
    p_sim ./= 4*T(π)*(1+η*ξ/2)
    return p_sim
end


Finite_corrections(α::T, ϵPE::T, ϵPA::T) where {T<:AbstractFloat} =
    (log(1/ϵPE)  + log(1/ϵPA))* α*inv(α-T(1)) - 2



function EC_cost_dmcv(L::Integer, f::T, N::T, pK::T, amplitude::T, ϵCR::T) where {T<:AbstractFloat}
    ξ = T(1)/100

    leak = hba_dmcv(L,ξ,amplitude)          # Conditional vN entropy
    leak *= f*pK                    # EC efficiency and pK
    leak += ceil(log2(1/ϵCR))/N     # Correctness cost
    return leak
end


function constraint_probabilities_dmcv(ρ::AbstractMatrix, Nc::Integer, Δs::T, Δ::T) where {T<:AbstractFloat}
    R_B = test_basis_dmcv(Nc,Δs,Δ)
    bases_AB = [kron(proj(x+1,4),R_B[z+1]) for x=0:3, z=0:5]
    return real(dot.(Ref(ρ),bases_AB))
end


function hbe_dmcv_general(
    L::Integer,
    N::T,
    Nc::Integer,
    pK::T,
    Δs::T,
    Δ ::T,
    ϵcompPE::T,
    amplitude::T,
    α::T;
    fast::Bool = true
) where {T<:AbstractFloat}


    dim_ρAB = 4 * (Nc + 1)
    model   = GenericModel{T}()

    # Variables
    @variable(model, ρAB[1:dim_ρAB, 1:dim_ρAB], Hermitian)
    @variable(model, q_K)
    @variable(model, q[1:4,1:6])
    @variable(model, h_QKD)
    @variable(model, h_KL)

    # Constraints on the marginal state
    ρA = partial_trace(ρAB, 2, [4, Nc+1])
    @constraint(model, ρA == alice_part(amplitude)) #this already implies tr(τAB) == 1

    # Constraints on probabilities
    @constraint(model, sum(q) + q_K == 1)

    # Constraints on exp vals via KL divergence
    p_ρAB = (1-pK)*constraint_probabilities_dmcv(ρAB,Nc,Δs,Δ)


    # Finite bounds via a Bretagnolle-Huber-Carol estimator
    δ = sqrt((2*25*log(2) - 2*log(ϵcompPE))/N)
    p_sim = simulated_probabilities_dmcv(Δs, Δ, amplitude, L)
    @constraint(model, [δ; q[:] - (1-pK)*p_sim[:];q_K - pK] in Hypatia.EpiNormInfCone{T,T}(1+1+length(q[:]),true))

    # Key map
    G    = gkraus(T,Nc)
    Ghat = [I(4*Nc+4)]
    Z    = zkraus(Nc)
    Zhat = [Zi*G for Zi in Z]


    # Reduction for block-diagonal structures
    permutation = vec(reshape(1:16*(Nc+1), 4, 4 * (Nc + 1))')
    Zhatperm = [Zi[permutation, :] for Zi ∈ Zhat]
    S = G[permutation, :]
    block_size = 4 * (Nc + 1)
    blocks = [(i-1)*block_size+1:i*block_size for i ∈ 1:4]

    vec_dim = Cones.svec_length(Complex, dim_ρAB)
    ρAB_vec = svec(ρAB)

    # Conic program
    @variable(model, u)
    if fast
        @constraint(model, [h_KL; vec(p_ρAB); vec(q)] in Hypatia.EpiRelEntropyCone{T}(1+2*length(q),false))
        β = inv(α)
        @constraint(
            model,
            [u; ρAB_vec] in EpiFastRenyiQKDTriCone{T,Complex{T}}(β, Ghat, Zhatperm, 1 + vec_dim; S, blocks)
        )
        sβ = β < 1 ? -1 : 1
        @constraint(model, [h_QKD, q_K, pK * sβ * u] in MOI.ExponentialCone())
        @objective(model, Min, α*inv(log(T(2))*(α-1)) * (h_KL - h_QKD))
    else
        @constraint(model, [h_KL; vec(p_ρAB); pK; vec(q); q_K] in Hypatia.EpiRelEntropyCone{T}(1+2+2*length(q),false))
        γ = inv(2 - inv(α))
        dim_σSAB = size(Z[1],2)
        @variable(model, σSAB[1:dim_σSAB, 1:dim_σSAB], Hermitian)
        @constraint(model, tr(σSAB) == 1)
        σSAB_vec = svec(σSAB)
        @constraint(
            model,
            [u; ρAB_vec; σSAB_vec] in EpiRenyiQKDTriCone{T,Complex{T}}(γ, Ghat, Z, 1 + length(ρAB_vec) + length(σSAB_vec); S=G, blocks)
        )
        sγ = γ < 1 ? -1 : 1
        @constraint(model, [h_QKD, 1, sγ * u] in MOI.ExponentialCone())
        @objective(model, Min, ((α / (α-1))*h_KL + (pK-δ)*h_QKD / (γ - 1))/log(T(2)))
    end


    # Optimize
    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    # Extract results
    h_renyi = objective_value(model)

    return h_renyi
end


function Finite_dmcv(L::Integer, f::T, N::T, Nc::Integer, pK::T, Δs::T, Δ::T; fast::Bool = true) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    # Pick the amplitude for the coherent states
    amplitude = optimal_amp(f, L)

    # Calculate EC cost per symbol
    leak_EC = EC_cost_dmcv(L, f, N, pK, amplitude, ϵCR)

    # Optimization wrt Renyi parameter α
    opt_renyi = T(1) + optimal_renyi(f,N,L)

    # If the optimal value for α is known, calculate the SKR
    if opt_renyi != 1
        correction = leak_EC + Finite_corrections(opt_renyi, ϵPE, ϵPA)/N

        # Conic program
        h_renyi = hbe_dmcv_general(L, N, Nc, pK, Δs, Δ, ϵcompPE, amplitude, opt_renyi; fast)

        SKR_Max = h_renyi - correction

        @printf("α-1 = %.5e, SKR = %.2e \n", opt_renyi-1, SKR_Max)
        
    # Otherwise, optimize with respect to α
    else
        finiteSKR_pars = Finite_pars(ϵPA, ϵPE, L, N, Nc, pK, Δs, Δ, ϵcompPE, amplitude, leak_EC, fast)
        optimize_renyi(α) = -FiniteSKR(α[1], finiteSKR_pars)

        # Initial guess
        α0 =[ T(1 +1e-4)]

        α_low = T(1)
        α_high = T(1.1) # A bit tightened, as our numerical analysis indicates
        options = Optim.Options(iterations = 100,f_calls_limit = 30)
        method  = Optim.NelderMead()
        sol = Optim.optimize(optimize_renyi, α_low, α_high, α0, method, options)
        opt_renyi = sol.minimizer[1]
        SKR_Max = -sol.minimum
    end

    return SKR_Max, opt_renyi, amplitude, leak_EC
end


# Suggested values for a test
# f = 1.0; N = 1e10; Nc = 5; Δs = 1.5; Δ = 4.0; T = Float64; L = 20; fast = true;
function Instance_dmcv(
    f::Real,
    N::Real,
    Nc::Integer;
    Δs::Real = 1.5,
    Δ::Real = 4.0,
    fast::Bool = true,
    T::DataType=Float64
    )

    ### Friendly reminder of basic parameters
    # ξ     = 0.01 (excess noise in SNUs)
    # α_att = 0.2  (attenuation at the fiber in dB/km)


    # Enforce desired precision
    f = T(f)
    N = T(N)
    Δs = T(Δs)
    Δ = T(Δ)

    # Create output file
    RATE_DMCV   = "Rate_dmcv_f"*string(Int(floor(f*100)))*"D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(Δs*10)))*".csv"
    FILE       = open(RATE_DMCV,"a")
    @printf(FILE,"xi, f, Nc, N, delta, Delta \n")
    @printf(FILE,"0.01, %.2f, %d, %.2f, %.2f, %.2f \n",f,Nc,log10(N),Δs,Δ)
    @printf(FILE,"L, amp, pK, a-1, leakEC, SKR \n")
    close(FILE)

    # Start loop for various values of the distance
    # Use threads for a speedup: Threads.@threads 
    for L ∈ 1:2:40

        # Pick the key round probability
        pK = optimal_pK(f, N, L)
        
        @printf("Distance: %d ---------\n",L)
        Finite_SKR, opt_renyi, amplitude, leak_EC  = Finite_dmcv(L, f, N, Nc, pK, Δs, Δ; fast)

        # Record outputs
        FILE = open(RATE_DMCV,"a")
        @printf(FILE,"%d, %.2f, %.2f, %.8e, %.12f, %.8e \n",L,amplitude,pK,opt_renyi-T(1),leak_EC,Finite_SKR)
        close(FILE)
    end
end
