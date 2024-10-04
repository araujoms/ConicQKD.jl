using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia
import Hypatia.Cones
import JLD2

"Produces a vector of `d` + 1 numerical MUBs for 2 ≤ `d` ≤ 13. For `d` = 6, 10, 12 the bases are
only roughly unbiased."
function numerical_mubs(d)
    mub_dict = JLD2.load("mubs.jld2")
    return mub_dict["mubs"][d]
end

"Decoherence map acting on Alice's key storage"
function zgmap(rho::AbstractMatrix, d::Integer)
    K = zgkraus(d)
    zgrho = sum(K[i] * rho * K[i] for i = 1:d)
    return Hermitian(zgrho)
end

function zgkraus(d::Integer)
    K = [kron(proj(i, d), I(d)) for i = 1:d]
    return K
end

"Computes a vector of probabilities that Alice and Bob get equal outcomes when measuring in bases C and C^T, respectively, where C is one ouf of `n` MUBs of dimension `d`"
function corr(
    ::Type{T},
    rho::AbstractMatrix,
    d::Integer,
    n::Integer;
    analytical_mub::Bool = false
) where {T<:AbstractFloat}
    if analytical_mub
        mubs = mub(Complex{T}, d) # analytical MUBs from the package Ket
    else
        mubs = numerical_mubs(d)
    end
    if T != Float64 && !analytical_mub
        @warn "To achieve higher precision analytical MUBs are needed."
    end
    b = [zeros(Complex{T}, d^2, d^2) for i = 1:n]
    for i = 1:n, j = 1:d
        temp = ketbra(mubs[i][:, j])
        b[i] += kron(temp, transpose(temp))
    end
    cleanup!.(b)
    b = Hermitian.(b)
    return real(dot.(Ref(rho), b))
end

"Computes the conditional entropy H(A|E) analytically for an isotropic state of dimension `d` with visibility `v`, using `n` MUBs. `d` must be a prime number, and `n` == 2 or `n` == `d` + 1"
function hae_mub_analytic(v::T, d::Integer, n::Integer = d + 1) where {T<:AbstractFloat}
    Q = 1 - v - (1 - v) / d
    if n == 2
        return log2(T(d)) + Q * log2(Q / (d - 1)) + (1 - Q) * log2(1 - Q)
    elseif n == d + 1
        return log2(T(d)) +
               (1 - (d + 1) * Q / d) * (log2(1 - Q - Q / d) - log2(1 - Q)) +
               (Q / d) * (log2(Q / (d^2 - d)) - log2(1 - Q)) +
               Q * log2(T(1) / d)
    else
        throw(ArgumentError("Number of MUBs must be either 2 or $(d+1), got $n."))
    end
end

rate_mub_analytic(v::Real, d::Integer, n::Integer = d + 1) = hae_mub_analytic(v, d, n) - hab_mub(v, d)

"Computes the conditional entropy H(A|B) for an isotropic state of dimension `d` with visibility `v`"
hab_mub(v::T, d) where {T<:AbstractFloat} = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(T(d) - 1)

"Computes the conditional entropy H(A|E) numerically for an isotropic state of dimension `d` with visibility `v`, using `n` MUBs. `n` must respect 2 ≤ `n` ≤ `d` + 1. `analytical_mub` specifies whether the MUBs are analytical or numerical."
function hae_mub(v::T, d::Integer, n::Integer = d + 1; analytical_mub::Bool = true) where {T<:AbstractFloat}
    is_complex = true
    model = GenericModel{T}()
    if is_complex
        @variable(model, ρ[1:d^2, 1:d^2], Hermitian)
        R = Complex{T}
    else
        @variable(model, ρ[1:d^2, 1:d^2], Symmetric)
        R = T
    end
    corr_ρ = corr(T, ρ, d, n; analytical_mub)
    W = v + (1 - v) / d
    corr_iso = W * ones(n)
    @constraint(model, corr_ρ .== corr_iso)
    @constraint(model, tr(ρ) == 1)

    vec_dim = Cones.svec_length(R, d^2)
    ρ_vec = svec(ρ, R)

    Ghat = [I(d^2)]
    Zhat = zgkraus(d)
    blocks = [(i-1)*d+1:i*d for i = 1:d]

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat, Zhat, 1 + vec_dim; blocks))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
    return solve_time(model)
end

rate_mub(v::T, d::Integer, n::Integer = d + 1; analytical_mub::Bool = false) where {T<:AbstractFloat} =
    hae_mub(v, d, n; analytical_mub) - hab_mub(v, d)
