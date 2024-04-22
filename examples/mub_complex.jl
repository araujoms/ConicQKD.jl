using ConicQKD
using JuMP
using Ket
using LinearAlgebra
import Hypatia
import Hypatia.Cones
import JLD2

"Generates complete set of MUBs for prime `d`"
function prime_mub(d)
    U = [zeros(ComplexF64, d, d) for _ = 1:d+1]
    U[1] = I(d)

    if d == 2
        U[2] = [1 1; 1 -1] / sqrt(2)
        U[3] = [1 1; im -im] / sqrt(2)
    else
        ω = exp(im * 2 * π / d)
        for k = 0:d-1, t = 0:d-1, j = 0:d-1
            exponent = mod(-j * t + k * div(j * (j - 1), 2), d)
            U[k+2][j+1, t+1] = ω^exponent / sqrt(d)
        end
    end

    cleanup!.(U)
    return U
end

function numerical_mub(d)
    mub_dict = JLD2.load("mubs.jld2")
    return mub_dict["mubs"][d]
end

"Decoherence map acting on Alice's key storage"
function zgmap(rho::AbstractMatrix, d::Integer)
    proj_comp = [ket(i, d) * ket(i, d)' for i = 1:d]
    K = [kron(proj_comp[i], I(d)) for i = 1:d]
    zgrho = sum(K[i] * rho * K[i] for i = 1:d)
    return Hermitian(zgrho)
end

function zgkraus(d::Integer, R)
    proj_comp = [proj(i, d; R) for i = 1:d]
    K = [kron(proj_comp[i], I(d)) for i = 1:d]
    return K
end

"Produces a vector of bases corresponding to the probabilities that Alice and Bob get equal outcomes when measuring in bases C and C^T, respectively, where C is one ouf of `n` MUBs of dimension `d`"
function bases(d::Integer, n::Integer)
    mub = numerical_mub(d)
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
function mub_rate_analytic(v::Real, d::Integer, n::Integer)
    Q = 1 - v - (1 - v) / d
    if n == 2
        return log2(d) + Q * log2(Q / (d - 1)) + (1 - Q) * log2(1 - Q)
    elseif n == d + 1
        return log2(d) + (1 - (d + 1) * Q / d) * (log2(1 - Q - Q / d) - log2(1 - Q)) + (Q / d) * (log2(Q / (d^2 - d)) - log2(1 - Q)) + Q * log2(1 / d)
    else
        error("n must be either 2 or d+1")
    end
end

"Computes the conditional entropy H(A|B) for an isotropic state of dimension `d` with visibility `v`"
hab(v, d) = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(d - 1)

"Computes the conditional entropy H(A|E) numerically for an isotropic state of dimension `d` with visibility `v`, using `n` MUBs.

Note that `d` must be a prime number, and 2 ≤ `n` ≤ `d` + 1"
function mub_rate(v::Real, d::Integer, n::Integer)
    T = Float64
    R = Complex{T}
    #    R = T
    is_complex = (R <: Complex)

    v = T(v)
    W = v + (1 - v) / d
    model = GenericModel{T}()
    if is_complex
        @variable(model, rho[1:d^2, 1:d^2], Hermitian)
    else
        @variable(model, rho[1:d^2, 1:d^2], Symmetric)
    end
    corr_rho = corr(rho, bases(d, n))
    corr_iso = W * ones(n)
    JuMP.@constraint(model, corr_rho .== corr_iso)
    JuMP.@constraint(model, tr(rho) == T(1))
    side::Int = size(rho, 2)
    vec_dim = Cones.svec_length(R, side)
    rho_vec = Vector{JuMP.AffExpr}(undef, vec_dim)

    if is_complex
        Cones._smat_to_svec_complex!(rho_vec, T(1) * rho, sqrt(T(2)))
    else
        Cones.smat_to_svec!(rho_vec, T(1) * rho, sqrt(T(2)))
    end

    G = [I(d^2)]
    ZG = zgkraus(d, R)

    JuMP.@variable(model, h)
    JuMP.@objective(model, Min, h / log(T(2)))
    JuMP.@constraint(model, [h; rho_vec] in EpiQKDTriCone{T,R}(G, ZG, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    JuMP.optimize!(model)
    return JuMP.objective_value(model)
    #    return JuMP.solve_time(model)    
end