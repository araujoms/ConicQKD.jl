using ConicQKD
using JuMP
using LinearAlgebra
using Ket
import Hypatia
import Hypatia.Cones

function zmap(ρ)
    K = zkraus()
    Zρ = sum(K[i] * ρ * K[i] for i = 1:2)
    return Zρ
end

function zkraus()
    K = [kron(proj(i, 2), I(2)) for i = 1:2]
    return K
end

function corr(ρ)
    Z = [proj(1, 2), proj(2, 2)]
    X = [0.5 * [1 1; 1 1], 0.5 * [1 -1; -1 1]]
    QZ = kron(Z[1], Z[2]) + kron(Z[2], Z[1])
    QX = kron(X[1], X[2]) + kron(X[2], X[1])
    global_basis = Hermitian.([QZ, QX])
    return real(dot.(Ref(ρ), global_basis))
end

function hae_bb84_general(qz::T, qx::T) where {T<:AbstractFloat}
    model = GenericModel{T}()
    dim_ρ = 4
    @variable(model, ρ[1:dim_ρ, 1:dim_ρ], Symmetric)
    corr_rho = corr(ρ)
    @constraint(model, corr_rho .== [qz, qx])
    @constraint(model, tr(ρ) == 1)

    Ghat = [I(dim_ρ)]
    Zhat = zkraus()

    vec_dim = Cones.svec_length(T, dim_ρ)
    ρ_vec = svec(ρ, T)

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    #return solve_time(model)
    return objective_value(model)
end

isometryz(::Type{T}) where {T} = [[1, 0, 0, 1] [1, 0, 0, -1]] / sqrt(T(2))
#isometryz2() = [[1,0,0,0] [0,0,0,1]]

function hae_bb84_reducedz(qx::T) where {T<:AbstractFloat}
    model = GenericModel{T}()
    dim_ρ = 2
    @variable(model, ρ[1:dim_ρ, 1:dim_ρ], Symmetric)
    @constraint(model, ρ[2, 2] == qx) #V'*QX*V == proj(2,2)
    @constraint(model, tr(ρ) == 1)

    V = isometryz(T)
    Ghat = [I(dim_ρ)]
    Z = zkraus()
    V2 = [[1, 0, 0, 0] [0, 0, 0, 1]]
    Zhat = [V2'Zi * V for Zi in Z]

    vec_dim = Cones.svec_length(T, dim_ρ)
    ρ_vec = svec(ρ, T)

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryx(::Type{T}) where {T<:AbstractFloat} = [[1, 0, 0, 1] [0, 1, 1, 0]] / sqrt(T(2))
#isometryx2() = 0.5 * [[1,1,1,1] [1,-1,-1,1]]

function hae_bb84_reducedx(qz::T) where {T<:AbstractFloat}
    model = GenericModel{T}()
    dim_ρ = 2
    @variable(model, ρ[1:dim_ρ, 1:dim_ρ], Symmetric)

    @constraint(model, ρ[2, 2] == qz) #V'*QZ*V == proj(2,2)
    @constraint(model, tr(ρ) == 1)

    V = isometryx(T)
    Ghat = [I(dim_ρ)]
    Z = zkraus()
    Zhat = [Zi * V for Zi in Z]

    vec_dim = Cones.svec_length(T, dim_ρ)
    ρ_vec = svec(ρ, T)

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

hae_bb84_analytic(qz, qx) = 1 - binary_entropy(qx)
hab_bb84(qz, qx) = binary_entropy(qz)
rate_bb84_analytic(qz, qx) = hae_bb84_analytic(qz, qx) - hab_bb84(qz, qx)

function hae_bb84(qz, qx)
    if qx == 0 && qz == 0
        return one(qx)
    elseif qz == 0
        return hae_bb84_reducedz(qx)
    elseif qx == 0
        return hae_bb84_reducedx(qz)
    else
        return hae_bb84_general(qz, qx)
    end
end

rate_bb84(qz, qx) = hae_bb84(qz, qx) - hab_bb84(qz, qx)
