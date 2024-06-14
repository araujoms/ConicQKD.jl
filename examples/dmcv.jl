using ConicQKD
using SpecialFunctions
using LinearAlgebra
using JuMP
using Ket
import Hypatia
import Hypatia.Cones
import Integrals

function integrand(vars, pars)
    γ = vars[1]
    θ = vars[2]
    ξ = pars[1]
    η = pars[2]
    x = pars[3]
    α = pars[4]
    return γ * exp(-abs2(γ * exp(im * θ) - sqrt(η) * im^x * α) / (1 + η * ξ / 2))
end

function integrate(bounds, pars)
    T = eltype(pars)
    problem = Integrals.IntegralProblem(integrand, bounds, pars)
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = Base.rtoldefault(T), abstol = Base.rtoldefault(T))
    return sol.u
end

function hba_dmcv(::Type{T}, L::Real, ξ::Real, α::Real) where {T}
    L, ξ, α = T.((L, ξ, α))
    η = 10^(-2 * L / 100)
    p_ab = zeros(T, 4, 4)
    for x = 0:3
        pars = [ξ, η, x, α]
        for z = 0:3
            bounds = ([T(0), T(π) * (2 * z - 1) / 4], [T(Inf), T(π) * (2 * z + 1) / 4])
            p_ab[x+1, z+1] = integrate(bounds, pars)
        end
    end
    p_ab ./= 4 * T(π) * (1 + η * ξ / 2)
    p_ba = transpose(p_ab)
    return conditional_entropy(p_ba)
end

function simulated_expectations(::Type{T}, L::Real, ξ::Real, α::Real) where {T}
    L, ξ, α = T.((L, ξ, α))
    η = 10^(-2 * L / 100)
    exp_sim = zeros(T, 4, 4)
    for x = 0:3
        exp_sim[x+1, 1] = sqrt(2 * η) * real(im^x * α)
        exp_sim[x+1, 2] = sqrt(2 * η) * imag(im^x * α)
        exp_sim[x+1, 3] = η * α^2 + η * ξ / 2
        exp_sim[x+1, 4] = 2 * η * (-1)^x * α^2
    end
    exp_sim ./= 4
    return exp_sim
end

function alice_part(α::Real)
    ρ = Hermitian(ones(Complex{typeof(α)}, 4, 4))
    ρ.data[1, 2] = exp(-(1 + im) * α^2)
    ρ.data[1, 3] = exp(-2 * α^2)
    ρ.data[1, 4] = exp(-(1 - im) * α^2)
    ρ.data[2, 3] = ρ.data[1, 2]
    ρ.data[2, 4] = ρ.data[1, 3]
    ρ.data[3, 4] = ρ.data[1, 2]
    ρ *= 0.25
end

function sinkpi4(::Type{T}, k::Integer) where {T} #computes sin(k*π/4) with high precision
    if mod(k, 4) == 0
        return 0
    else
        signal = (-1)^div(k, 4, RoundDown)
        if mod(k, 2) == 0
            return signal
        else
            return signal / sqrt(T(2))
        end
    end
end

function region_operators(::Type{T}, Nc::Integer) where {T}
    R = [Hermitian(zeros(Complex{T}, Nc + 1, Nc + 1)) for z = 0:3]
    for z = 0:2
        for n = 0:Nc
            for m = n:Nc
                if n == m
                    R[z+1][n+1, m+1] = T(1) / 4
                else
                    angular = 2 * im^(mod(z * (n - m), 4)) * sinkpi4(T, n - m) / (n - m)
                    radial = gamma(1 + T(n + m) / 2) / (2 * T(π) * sqrt(gamma(T(1 + n)) * gamma(T(1 + m))))
                    R[z+1].data[n+1, m+1] = angular * radial
                end
            end
        end
    end
    R[4] = I - sum(R[z+1] for z = 0:2)
    return R
end

function annihilation_operator(::Type{T}, Nc::Integer) where {T}
    dl = zeros(T, Nc)
    d = zeros(T, Nc + 1)
    du = sqrt.(T.(1:Nc))
    return Tridiagonal(dl, d, du)
end

function heterodyne_operators(::Type{T}, Nc::Integer) where {T}
    a = annihilation_operator(T, Nc)
    q = (a' + a) / sqrt(T(2))
    p = im * (a' - a) / sqrt(T(2))
    n = Diagonal(T.(0:Nc))
    d = a^2 + (a')^2
    return [q, p, n, d]
end

function constraint_expectations(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T}
    ops = heterodyne_operators(T, Nc)
    bases_AB = [kron(proj(x + 1, 4), ops[z+1]) for x = 0:3, z = 0:3]
    return real(dot.(Ref(ρ), bases_AB))
end

function gmap(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T}
    V = gkraus(T, Nc)
    return Hermitian(V * ρ * V')
end

function gkraus(::Type{T}, Nc::Integer) where {T}
    sqrtbasis = sqrt.(region_operators(T, Nc))
    V = sum(kron(I(4), sqrtbasis[i], ket(i, 4)) for i = 1:4)
    return V
end

function zmap(ρ::AbstractMatrix, Nc::Integer, T::DataType = Float64)
    K = zkraus(Nc)
    return Hermitian(sum(K[i] * ρ * K[i] for i = 1:4))
end

function zkraus(Nc::Integer)
    K = [kron(I(4 * (Nc + 1)), proj(i, 4)) for i = 1:4]
    return K
end

function rate_dmcv(::Type{T}, Nc::Integer, L::Real, ξ::Real, α::Real) where {T}
    return hbe_dmcv(T, Nc, L, ξ, α) - hba_dmcv(T, L, ξ, α)
end
rate_dmcv(Nc::Integer, L::Real, ξ::Real, α::Real) = rate_dmcv(Float64, Nc, L, ξ, α)

function hbe_dmcv(::Type{T}, Nc::Integer, L::Real, ξ::Real, α::Real) where {T}
    L, ξ, α = T.((L, ξ, α))
    dim_ρAB = 4 * (Nc + 1)
    model = GenericModel{T}()
    @variable(model, ρAB[1:dim_ρAB, 1:dim_ρAB], Hermitian)

    exp_ρAB = constraint_expectations(T, ρAB, Nc)
    exp_sim = simulated_expectations(T, L, ξ, α)
    @constraint(model, exp_sim .== exp_ρAB)
    ρA = partial_trace(T(1) * ρAB, 2, [4, Nc + 1])
    @constraint(model, ρA == alice_part(α))

    G = gkraus(T, Nc)
    Ghat = [I(dim_ρAB)]
    Z = zkraus(Nc)
    Zhat = [Zi * G for Zi in Z]
    permutation = vec(reshape(1:16*(Nc+1), 4, 4 * (Nc + 1))')
    Zhatperm = [Zi[permutation, :] for Zi in Zhat]

    block_size = 4 * (Nc + 1)
    blocks = [(i-1)*block_size+1:i*block_size for i = 1:4]

    vec_dim = Cones.svec_length(Complex, dim_ρAB)
    ρAB_vec = Vector{GenericAffExpr{T,GenericVariableRef{T}}}(undef, vec_dim)
    Cones._smat_to_svec_complex!(ρAB_vec, ρAB, sqrt(T(2)))

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρAB_vec] in EpiQKDTriCone{T,Complex{T}}(Ghat, Zhatperm, 1 + vec_dim; blocks))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
    return solve_time(model)
end
hbe_dmcv(Nc::Integer, L::Real, ξ::Real, α::Real) = hbe_dmcv(Float64, Nc, L, ξ, α)
hbe_dmcv(Nc::Integer) = hbe_dmcv(Nc, 60, 0.05, 0.35)
