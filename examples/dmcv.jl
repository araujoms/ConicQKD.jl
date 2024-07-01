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
    tol = T == Float64 ? eps(T)^(3 / 4) : sqrt(eps(T))
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = tol, abstol = tol)
    return sol.u
end

function joint_probability(L::T, ξ::T, α::T) where {T<:AbstractFloat}
    η = 10^(-2 * L / 100)
    pAB = zeros(T, 4, 4)
    for x = 0:3
        pars = [ξ, η, x, α]
        for z = 0:3
            bounds = ([T(0), T(π) * (2 * z - 1) / 4], [T(Inf), T(π) * (2 * z + 1) / 4])
            pAB[x+1, z+1] = integrate(bounds, pars)
        end
    end
    pAB ./= 4 * T(π) * (1 + η * ξ / 2)
    return pAB
end

function hba_dmcv(L::T, ξ::T, α::T) where {T<:AbstractFloat}
    pAB = joint_probability(L, ξ, α)
    pBA = transpose(pAB)
    return conditional_entropy(pBA)
end

function hbe_dmcv_analytic(L::T, α::T) where {T<:AbstractFloat}
    η = 10^(-2 * L / 100)
    c =
        exp(-(1 - η) * α^2 / 2) / sqrt(T(2)) * [
            sqrt(cosh((1 - η) * α^2) + cos((1 - η) * α^2)),
            sqrt(sinh((1 - η) * α^2) + sin((1 - η) * α^2)),
            sqrt(cosh((1 - η) * α^2) - cos((1 - η) * α^2)),
            sqrt(sinh((1 - η) * α^2) - sin((1 - η) * α^2))
        ]
    eve_states = [[c[j+1] * exp(im * T(π) * i * j / 2) for j = 0:3] for i = 0:3]
    pAB = joint_probability(L, T(0), α)
    pA = sum(pAB; dims = 2)
    ρE = sum(pA[x] * ketbra(eve_states[x]) for x = 1:4)
    ρBE = sum(pAB[x, z] * kron(proj(z, 4), ketbra(eve_states[x])) for x = 1:4, z = 1:4)
    return entropy(ρBE) - entropy(ρE)
end

function simulated_expectations(L::T, ξ::T, α::T) where {T<:Real}
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

function alice_part(α::T) where {T<:Real}
    ρ = zeros(Complex{T}, 4, 4)
    for j = 0:3, i = 0:j
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

function region_operators(::Type{T}, Nc::Integer) where {T<:Real}
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

function annihilation_operator(::Type{T}, Nc::Integer) where {T<:Real}
    dl = zeros(T, Nc)
    d = zeros(T, Nc + 1)
    du = sqrt.(T.(1:Nc))
    return Tridiagonal(dl, d, du)
end

function heterodyne_operators(::Type{T}, Nc::Integer) where {T<:Real}
    a = annihilation_operator(T, Nc)
    q = (a' + a) / sqrt(T(2))
    p = im * (a' - a) / sqrt(T(2))
    n = Diagonal(T.(0:Nc))
    d = a^2 + (a')^2
    return [q, p, n, d]
end

function constraint_expectations(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T<:Real}
    ops = heterodyne_operators(T, Nc)
    bases_AB = [kron(proj(x + 1, 4), ops[z+1]) for x = 0:3, z = 0:3]
    return real(dot.(Ref(ρ), bases_AB))
end

function gmap(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T}
    V = gkraus(T, Nc)
    return Hermitian(V * ρ * V')
end

function gkraus(::Type{T}, Nc::Integer) where {T<:Real}
    sqrtbasis = sqrt.(region_operators(T, Nc))
    #    cleanup!.(sqrtbasis;tol=10^3*eps(T))
    V = sum(kron(I(4), sqrtbasis[i], ket(i, 4)) for i = 1:4)
    return V
end

function zmap(ρ::AbstractMatrix, Nc::Integer)
    K = zkraus(Nc)
    return Hermitian(sum(K[i] * ρ * K[i] for i = 1:4))
end

function zkraus(Nc::Integer)
    K = [kron(I(4 * (Nc + 1)), proj(i, 4)) for i = 1:4]
    return K
end

function hbe_dmcv_general(Nc::Integer, L::T, ξ::T, α::T) where {T<:AbstractFloat}
    dim_ρAB = 4 * (Nc + 1)
    model = GenericModel{T}()
    @variable(model, ρAB[1:dim_ρAB, 1:dim_ρAB], Hermitian)

    exp_ρAB = constraint_expectations(T, ρAB, Nc)
    exp_sim = simulated_expectations(L, ξ, α)
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
    ρAB_vec = svec(ρAB, Complex{T})

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρAB_vec] in EpiQKDTriCone{T,Complex{T}}(Ghat, Zhatperm, 1 + vec_dim; blocks))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
    return solve_time(model)
end

coherent(Nc::Integer, β::Number) = exp(-abs2(β) / 2) * [β^n / sqrt(factorial(n)) for n = 0:Nc]
isometry(Nc::Integer, α::Real) = sum(kron(ket(x + 1, 4), coherent(Nc, im^x * α)) * ket(x + 1, 4)' for x = 0:3)
function hbe_dmcv_reduced(Nc::Integer, L::T, α::T) where {T<:AbstractFloat}
    dim_σAB = 4
    model = GenericModel{T}()

    η = 10^(-2 * L / 100)
    σAB = Matrix{Complex{T}}(undef, 4, 4)
    for x2 = 0:3, x1 = 0:3
        σAB[x1+1, x2+1] = 0.25 * exp(-α^2 * (1 - η) * (1 - (1.0 * im)^(x1 - x2)))
    end
    σAB = Hermitian(σAB)

    #    V = isometry(Nc,sqrt(η)*α)
    #    ρAB = Hermitian(V*σAB*V')

    sqrtbasis = sqrt.(region_operators(T, Nc))
    states = [sqrtbasis[k] * coherent(Nc, im^x * sqrt(η) * α) for k = 1:4, x = 0:3]
    norms = norm.(states)

    Ghat = [I(dim_σAB)]
    Zhat = [sum(norms[k, x] * kron(proj(x, 4), ket(k, 4)) for x = 1:4) for k = 1:4]

    permutation = vec(reshape(1:16, 4, 4)')
    Zhatperm = [Zi[permutation, :] for Zi in Zhat]

    block_size = 4
    blocks = [(i-1)*block_size+1:i*block_size for i = 1:4]

    vec_dim = Cones.svec_length(Complex, dim_σAB)
    σAB_vec = svec(σAB, Complex{T})

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; σAB_vec] in EpiQKDTriCone{T,Complex{T}}(Ghat, Zhatperm, 1 + vec_dim; blocks))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
    return solve_time(model)
end

function hbe_dmcv(Nc::Integer, L::T, ξ::T, α::T) where {T<:AbstractFloat}
    if ξ == 0
        return hbe_dmcv_reduced(Nc, L, α)
    else
        return hbe_dmcv_general(Nc, L, ξ, α)
    end
end
hbe_dmcv(Nc::Integer) = hbe_dmcv(Nc, 60.0, 0.05, 0.35)

function rate_dmcv_analytic(L::T, ξ::T, α::T) where {T<:AbstractFloat}
    return hbe_dmcv_analytic(L, α) - hba_dmcv(L, ξ, α)
end

function rate_dmcv(Nc::Integer, L::T, ξ::T, α::T) where {T<:AbstractFloat}
    return hbe_dmcv(Nc, L, ξ, α) - hba_dmcv(L, ξ, α)
end

#f64 2.801905307769914e-8
#d64 6.03065794378403782333344081012188704e-14
