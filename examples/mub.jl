using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia
import Hypatia.Cones
import JLD2

function analytical_mubs(d)
    return mub(d)
end

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

"Produces a vector of bases corresponding to the probabilities that Alice and Bob get equal outcomes when measuring in bases C and C^T, respectively, where C is one ouf of `n` MUBs of dimension `d`"
function bases(d::Integer, n::Integer)
    mub = numerical_mubs(d)
    b = [zeros(ComplexF64, d^2, d^2) for i = 1:n]
    for i = 1:n, j = 1:d
        temp = ketbra(mub[i][:, j])
        b[i] += kron(temp, transpose(temp))
    end
    cleanup!.(b)
    return Hermitian.(b)
end

"Computes vector of probabilities of quantum state `rho` in bases `bases`"
corr(rho::AbstractMatrix, bases::AbstractVector) = real(dot.(Ref(rho), bases))

"Computes the conditional entropy H(A|E) analytically for an isotropic state of dimension `d` with visibility `v`, using `n` MUBs.

Note that `d` must be a prime number, and `n` == 2 or `n` == `d` + 1"
function hae_mub_analytic(v::Real, d::Integer, n::Integer = d + 1)
    Q = 1 - v - (1 - v) / d
    if n == 2
        return log2(d) + Q * log2(Q / (d - 1)) + (1 - Q) * log2(1 - Q)
    elseif n == d + 1
        return log2(d) +
               (1 - (d + 1) * Q / d) * (log2(1 - Q - Q / d) - log2(1 - Q)) +
               (Q / d) * (log2(Q / (d^2 - d)) - log2(1 - Q)) +
               Q * log2(1 / d)
    else
        error("n must be either 2 or d+1")
    end
end

rate_mub_analytic(v::Real, d::Integer, n::Integer = d + 1) = hae_mub_analytic(v, d, n) - hab_mub(v, d)

rate_mub(::Type{T}, v::Real, d::Integer, n::Integer = d + 1) where {T} = hae_mub(T, v, d, n) - hab_mub(v, d)
rate_mub(v::Real, d::Integer, n::Integer = d + 1) = rate_mub(ComplexF64, v, d, n)

"Computes the conditional entropy H(A|B) for an isotropic state of dimension `d` with visibility `v`"
hab_mub(v, d) = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(d - 1)

"Computes the conditional entropy H(A|E) numerically for an isotropic state of dimension `d` with visibility `v`, using `n` MUBs.

Note that `d` must be a prime number, and 2 ≤ `n` ≤ `d` + 1"
function hae_mub(::Type{T}, v::Real, d::Integer, n::Integer = d + 1) where {T}
    R = real(T)
    is_complex = (T <: Complex)
    v = R(v)
    model = GenericModel{R}()
    if is_complex
        @variable(model, rho[1:d^2, 1:d^2], Hermitian)
    else
        @variable(model, rho[1:d^2, 1:d^2], Symmetric)
    end
    corr_rho = corr(rho, bases(d, n))
    W = v + (1 - v) / d
    corr_iso = W * ones(n)
    @constraint(model, corr_rho .== corr_iso)
    @constraint(model, tr(rho) == 1)
    vec_dim = Cones.svec_length(T, d^2)
    rho_vec = Vector{GenericAffExpr{R,GenericVariableRef{R}}}(undef, vec_dim)

    if is_complex
        Cones._smat_to_svec_complex!(rho_vec, T(1) * rho, sqrt(R(2)))
    else
        Cones.smat_to_svec!(rho_vec, T(1) * rho, sqrt(R(2)))
    end

    G = [I(d^2)]
    ZG = zgkraus(d)
    blocks = [(i-1)*d+1:i*d for i = 1:d]

    @variable(model, h)
    @objective(model, Min, h / log(R(2)))
    @constraint(model, [h; rho_vec] in EpiQKDTriCone{R,T}(G, ZG, 1 + vec_dim; blocks))

    set_optimizer(model, Hypatia.Optimizer{R})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end
hae_mub(v::Real, d::Integer, n::Integer = d + 1) = hae_mub(ComplexF64, v, d, n)
