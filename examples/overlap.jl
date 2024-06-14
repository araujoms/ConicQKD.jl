using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia
import Hypatia.Cones

function zgmap(rho::AbstractMatrix, d::Integer)
    K = zgkraus(d)
    zgrho = sum(K[i] * rho * K[i] for i = 1:d)
    return zgrho
end

function zgkraus(d::Integer)
    K = [kron(proj(i, d), I(d)) for i = 1:d]
    return K
end

function local_bases(::Type{T}, d::Integer) where {T}
    localb = Vector{Vector{Hermitian{T,Matrix{T}}}}(undef, d == 2 ? 2 : 3)

    localb[1] = [ketbra(ket(i, d)) for i = 1:d]

    localb[2] = Vector{Hermitian{T,Matrix{T}}}(undef, d)
    for i = 1:div(d, 2)
        localb[2][2*i-1] = ketbra(ket(2 * i - 1, d) + ket(2 * i, d)) / 2
        localb[2][2*i] = ketbra(ket(2 * i - 1, d) - ket(2 * i, d)) / 2
    end
    if mod(d, 2) == 1
        localb[2][d] = ketbra(ket(d, d))
    end

    if d >= 3
        localb[3] = Vector{Hermitian{T,Matrix{T}}}(undef, d)
        localb[3][1] = ketbra(ket(1, d))
        for i = 1:div(d - 1, 2)
            localb[3][2*i] = ketbra(ket(2 * i, d) + ket(2 * i + 1, d)) / 2
            localb[3][2*i+1] = ketbra(ket(2 * i, d) - ket(2 * i + 1, d)) / 2
        end
        if mod(d, 2) == 0
            localb[3][d] = ketbra(ket(d, d))
        end
    end
    return localb
end

function bases_equal(::Type{T}, d::Integer) where {T}
    localb = local_bases(T, d)

    b = Vector{Hermitian{T,Matrix{T}}}(undef, length(localb) * d)
    counter = 0
    for k = 1:length(localb), i = 1:d
        counter += 1
        b[counter] = Hermitian(kron(localb[k][i], transpose(localb[k][i])))
    end
    return b
end

function bases_full(::Type{T}, d::Integer) where {T}
    localb = local_bases(T, d)
    num_indep = div(5 * d^2 - 2 * d - 3, 2)
    b = Vector{Hermitian{T,Matrix{T}}}(undef, num_indep)
    counter = 0
    goodindices = [[i for i = 1:d-1], [i for i = 1:2:d-1], [i for i = 2:2:d-1]]
    for k = 1:length(localb)
        for i = 1:d, j = 1:d
            if i in goodindices[k] || j in goodindices[k]
                counter += 1
                b[counter] = Hermitian(kron(localb[k][i], transpose(localb[k][j])))
            end
        end
    end
    return b
end

"Computes vector of probabilities of quantum state `rho` in bases `bases`"
corr(rho::AbstractMatrix, bases::AbstractVector) = real(dot.(Ref(rho), bases))

rate_overlap(::Type{T}, v::Real, d::Integer) where {T} = hae_overlap(T, v, d) - hab_overlap(v, d)
rate_overlap(v::Real, d::Integer) = rate_overlap(Float64, v, d)

hab_overlap(v, d) = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(d - 1)

function hae_overlap(::Type{T}, v::Real, d::Integer) where {T}
    R = real(T)
    is_complex = (T <: Complex)

    model = GenericModel{R}()
    if is_complex
        @variable(model, rho[1:d^2, 1:d^2], Hermitian)
    else
        @variable(model, rho[1:d^2, 1:d^2], Symmetric)
    end
    bases = bases_full(T, d)
    corr_rho = corr(rho, bases)
    corr_iso = corr(isotropic(v, d), bases)
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
    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end
hae_overlap(v::Real, d::Integer) = hae_overlap(Float64, v, d)
