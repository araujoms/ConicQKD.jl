using LinearAlgebra
using JuMP
using ConicQKD
using Ket
import Hypatia

function zkraus(d::Integer)
    K = [kron(proj(i, d), I(d)) for i ∈ 1:d]
    return K
end

function local_bases(::Type{T}, d::Integer) where {T}
    localb = Vector{Vector{Hermitian{T,Matrix{T}}}}(undef, d == 2 ? 2 : 3)

    localb[1] = [ketbra(ket(i, d)) for i ∈ 1:d]

    localb[2] = Vector{Hermitian{T,Matrix{T}}}(undef, d)
    for i ∈ 1:div(d, 2)
        localb[2][2*i-1] = ketbra(ket(2 * i - 1, d) + ket(2 * i, d)) / 2
        localb[2][2*i] = ketbra(ket(2 * i - 1, d) - ket(2 * i, d)) / 2
    end
    if mod(d, 2) == 1
        localb[2][d] = ketbra(ket(d, d))
    end

    if d >= 3
        localb[3] = Vector{Hermitian{T,Matrix{T}}}(undef, d)
        localb[3][1] = ketbra(ket(1, d))
        for i ∈ 1:div(d - 1, 2)
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
    for k ∈ 1:length(localb), i ∈ 1:d
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
    goodindices = [[i for i ∈ 1:d-1], [i for i ∈ 1:2:d-1], [i for i ∈ 2:2:d-1]]
    for k ∈ 1:length(localb)
        for i ∈ 1:d, j ∈ 1:d
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

hab_overlap(v::T, d) where {T<:AbstractFloat} = binary_entropy(v + (1 - v) / d) + (1 - v - (1 - v) / d) * log2(T(d) - 1)

function hae_overlap(v::T, d::Integer, α::T = T(11) / 10; renyi = false, fast = true) where {T<:AbstractFloat}
    model = GenericModel{T}()
    @variable(model, ρ[1:d^2, 1:d^2], Symmetric)
    bases = bases_full(T, d)
    corr_ρ = corr(ρ, bases)
    corr_iso = corr(state_phiplus(T, d; v), bases)
    @constraint(model, corr_ρ .== corr_iso)
    @constraint(model, tr(ρ) == 1)

    ρ_vec = svec(ρ)
    vec_dim = length(ρ_vec)

    Ghat = [I(d^2)]
    Zhat = zkraus(d)
    blocks = [(i-1)*d+1:i*d for i ∈ 1:d]

    @variable(model, h)
    @objective(model, Min, h)
    if renyi
        if fast
            β = inv(α)
            @constraint(model, [h; ρ_vec] in EpiFastRenyiQKDTriCone{T,T}(β, Ghat, Zhat, 1 + vec_dim; blocks))
        else
            @variable(model, σ[1:d^2, 1:d^2], Symmetric)
            @constraint(model, tr(σ) == 1)
            σ_vec = svec(σ)
            β = inv(2 - inv(α))
            @constraint(model, [h; ρ_vec; σ_vec] in EpiRenyiQKDTriCone{T,T}(β, Ghat, Zhat, 1 + 2vec_dim; blocks))
        end
    else
        @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,T}(Ghat, Zhat, 1 + vec_dim; blocks))
    end

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    if renyi
        sβ = β < 1 ? -1 : 1
        return log2(sβ * value(h)) / (β - 1)
    else
        return value(h) / log(T(2))
    end
    return objective_value(model)
end

rate_overlap(v::T, d::Integer) where {T<:AbstractFloat} = hae_overlap(v, d) - hab_overlap(v, d)
