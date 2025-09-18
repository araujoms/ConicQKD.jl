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

# import MOI: ExponentialCone


function alice_part(γ::Real)
    ρ = Hermitian(ones(Complex{typeof(γ)},4,4))
    ρ.data[1,2] = (exp(-(1+im)*γ^2))
    ρ.data[1,3] = (exp(-2*γ^2))
    ρ.data[1,4] = (exp(-(1-im)*γ^2))
    ρ.data[2,3] = ρ.data[1,2]
    ρ.data[2,4] = ρ.data[1,3]
    ρ.data[3,4] = ρ.data[1,2]
    ρ *= 0.25
end


function integrand(vars,pars)
    ζ = vars[1]
    θ = vars[2]
    ξ = pars[1]
    η = pars[2]
    x = pars[3]
    γ = pars[4]

    return ζ*exp(-abs2(ζ*exp(im*θ)-sqrt(η)*im^x*γ)/(1+η*ξ/2))
end

function integrate(bounds, pars)
    T = eltype(pars)
    problem = Integrals.IntegralProblem(integrand, bounds, pars)
    tol = T == Float64 ? eps(T) : sqrt(eps(T))
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = tol, abstol = tol)
    return sol.u
end

function sinkpi4(::Type{T}, k::Integer) where {T<:AbstractFloat}
    if mod(k,4) == 0
        return T(0)
    else
        signal = T((-1)^div(k,4,RoundDown))
        if mod(k,2) == 0
            return signal
        else
            return signal/sqrt(T(2))
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

@with_kw struct epsilon_coeffs{T<:AbstractFloat}
    ϵCR::T = 1e-11
    ϵPA::T = 9e-11
    ϵPE::T = 9e-11
    ϵcompPE::T = 9e-11
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


function simulated_probabilities_dmcv(Δs::T, Δ::T, γ::T, D::Integer) where {T<:AbstractFloat}
    α_att = T(2)/10
    ξ = T(1)/100
    η = 10^(-(α_att*D)/10)
    p_sim = zeros(T,4,6)
    for x=0:3
        pars = [ξ, η, x, γ]
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
    (log(1/ϵPE)  + log(1/ϵPA))* α/(α-T(1)) - 2


function EC_cost_dmcv(D::Integer, f::T, N::T, pK::T, γ::T, ϵCR::T) where {T<:AbstractFloat}
    
    α_att = T(2)/10
    ξ     = T(1)/100
    η     = 10^(- α_att*D/10)
    p_EC  = zeros(T,4,4)               # Conditional probability p(z|x)

    # Calculate p_EC(z|x) = p_EC(z,x)/4
    for x=0:3
        pars = [ξ, η, x, γ]
        for z=0:3
            bounds        = ([T(0),T(π)*(2*z-1)/4],[T(Inf),T(π)*(2*z+1)/4])
            p_EC[x+1,z+1] = integrate(bounds,pars)
        end
    end 
    p_EC /= T(π)*(1 + η*ξ/2)

    # Renormalize the distribution
    p_PS  = sum(p_EC/4)
    p_EC /= p_PS

    leak = -p_EC[:]'*log2.(p_EC[:])*T(0.25) # Conditional vN entropy
    leak *= f*pK                            # EC efficiency and pK
    leak += ceil(log2(1/ϵCR))/N             # Correctness cost
    return leak
end


function constraint_probabilities_dmcv(ρ::AbstractMatrix, Nc::Integer, Δs::T, Δ::T) where {T<:AbstractFloat}
    R_B = test_basis_dmcv(Nc,Δs,Δ)
    bases_AB = [kron(proj(x+1,4),R_B[z+1]) for x=0:3, z=0:5]
    return real(dot.(Ref(ρ),bases_AB))
end

function conic_dmcv(D::Integer, N::T, pK::T, Nc::Integer, Δs::T, Δ::T, ϵcompPE::T, γ::T, α::T; renyi::Bool = true) where {T<:AbstractFloat}
    model = GenericModel{T}()
    R = Complex{T}

    dim_ρAB = 4*(Nc+1)

    # Variables
    @variable(model, ρAB[1:dim_ρAB, 1:dim_ρAB], Hermitian)
    @variable(model, q_K ≥ 0)
    @variable(model, q[1:4,1:6] ≥ 0)

    # Constraints on the marginal state
    ρA = partial_trace(ρAB, 2, [4, Nc+1])
    @constraint(model, ρA == alice_part(γ)) #this already implies tr(τAB) == 1

    # Constraints on exp vals via KL divergence
    p_ρAB = (1-pK)*constraint_probabilities_dmcv(ρAB,Nc,Δs,Δ)
    @variable(model, h_KL)
    @constraint(model, [h_KL; p_ρAB[:];pK; q[:];q_K] in Hypatia.EpiRelEntropyCone{T}(1+2+2*length(q[:]),false))

    # Constraints on probabilities
    @constraint(model, sum(q) + q_K == 1)

    # Finite bounds via a Bretagnolle-Huber-Carol estimator
    δ = sqrt((2*25*log(2) - 2*log(ϵcompPE))/N)
    p_sim = simulated_probabilities_dmcv(Δs, Δ, γ, D)
    @constraint(model, [δ; q[:] - (1-pK)*p_sim[:];q_K - pK] in Hypatia.EpiNormInfCone{T,T}(1+1+length(q[:]),true))

    # Key map
    G    = gkraus(T,Nc)
    Ghat = [I(4*Nc+4)]
    Z    = zkraus(Nc)
    Zhat = [Zi*G for Zi in Z]

    permutation = vec(reshape(1:16*(Nc+1),4,4*(Nc+1))')
    Zhatperm    = [Zi[permutation,:] for Zi in Zhat]
    S           = G[permutation,:]
    block_size  = 4*(Nc+1)
    blocks      = [(i-1)*block_size+1:i*block_size for i=1:4]
    
    vec_dim = Cones.svec_length(Complex,dim_ρAB)
    ρ_vec = svec(ρAB)

    # QKD (Rényi) cone 
    @variable(model, Ψ)
    if renyi
        @variable(model, h)
        β = inv(2 - inv(α))
        sβ = β < 1 ? -1 : 1
        @constraint(model, [Ψ; ρ_vec] in EpiRenyiQKDTriCone{T,R}(β, Ghat, Zhatperm, 1 + vec_dim; S, blocks))
        @constraint(model, [h * (β - 1), 1, sβ * Ψ] in MOI.ExponentialCone())
        @objective(model, Min, α*inv(log(T(2))*(α-T(1)))*h_KL + (pK-δ)*inv(log(T(2)))*h)
    else
        throw("Not implemented yet")
        # @constraint(model, [Ψ; ρ_vec] in EpiQKDTriCone{T,R}(Ghat, Zhatperm, 1 + vec_dim; blocks))
    end

    
    # Optimize
    # @objective(model, Min, α*inv(log(T(2))*(α-T(1)))*h_KL + (pK-δ)*h_α) # If q_k fails, use (pK - δ)
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


function Finite_dmcv(D::Integer, f::T, N::T, pK::T, Nc::Integer, Δs::T, Δ::T; renyi::Bool = false) where {T<:AbstractFloat}

    # Load the epsilons
    @unpack ϵCR, ϵPA, ϵPE, ϵcompPE = epsilon_coeffs{T}()

    # Pick amplitude for the coherent states
    γ = D == 20 ? 0.77 : 0.8

    # Calculate EC cost per symbol
    leak_EC = EC_cost_dmcv(D, f, N, pK, γ, ϵCR)

    """ Here I need an optimization wrt α """ 
    α = D == 20 ? T(1 +1.911e-5) : T(1 +5e-4) # Test value
    # Total correction
    correction = leak_EC + Finite_corrections(α, ϵPE, ϵPA)/N

    # Conic program
    h_renyi = conic_dmcv(D, N, pK, Nc, Δs, Δ, ϵcompPE, γ, α ; renyi)

    Finite_SKR = h_renyi - correction
    return Finite_SKR, α, γ, leak_EC
end

# f = 1.0; N = 1e10; pK = 0.73; Nc = 5; Δs = 1.5; Δ = 4.0; T = Float64; D = 20; renyi = true;
function Instance_dmcv(
    f::Real,
    N::Real,
    pK::Real,
    Nc::Integer;
    Δs::Real = 1.5,
    Δ::Real = 4.0,
    T::DataType=Float64
    )

    ### Friendly reminder of basic parameters
    # ξ     = 0.01 (excess noise in SNUs)
    # α_att = 0.2  (attenuation at the fiber in dB/km)


    # Enforce desired precision
    f = T(f)
    N = T(N)
    pK = T(pK)
    Δs = T(Δs)
    Δ = T(Δ)

    # Create output file
    RATE_DMCV   = "Rate_dmcv_f"*string(Int(floor(f*100)))*"D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(Δs*10)))*".csv"
    FILE       = open(RATE_DMCV,"a")
    @printf(FILE_RATE,"xi, f, Nc, N, delta, Delta, nu, eta \n")
    @printf(FILE_RATE,"0.01, %.2f, %d, %.2f, %.2f, %.2f \n",f,Nc,log10(N),Δs,Δ)
    @printf(FILE_RATE,"D, amp, pK, a-1, leakEC, SKR \n")
    close(FILE)

    # Start loop for various values of the distance
    # Threads.@threads 
    for D ∈ 1:5:40
        @printf("Distance: %d ---------\n",D)
        Finite_SKR, optimal_α, γ, leak_EC  = Finite_dmcv(D, f, N, pK, Nc, Δs, Δ; renyi = true)

        # Record outputs
        FILE = open(RATE_DMCV,"a")
        @printf(FILE_RATE,"%d, %.2f, %.2f, %.8e, %.12f, %.8e \n",D,γ,pK,optimal_α-T(1),leak_EC,Finite_SKR)
        close(FILE)
    end
end