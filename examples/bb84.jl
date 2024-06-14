using ConicQKD
using JuMP
using LinearAlgebra
using Ket
import Hypatia
import Hypatia.Cones

function gmap(ρ, p_z, η0, η1; T = Float64)
    Z = [ket(1, 2; T) * ket(1, 2; T)', ket(2, 2; T) * ket(2, 2; T)']
    X = [Complex{T}(0.5) * [1 1; 1 1], Complex{T}(0.5) * [1 -1; -1 1]]
    ZB = [zeros(Complex{T}, 2, 2) for i = 1:2]
    XB = [zeros(Complex{T}, 2, 2) for i = 1:2]
    ZB[1][1:2, 1:2] = sqrt(η0) * Z[1]
    ZB[2][1:2, 1:2] = sqrt(η1) * Z[2]
    XB[1][1:2, 1:2] = sqrt(η0) * X[1]
    XB[2][1:2, 1:2] = sqrt(η1) * X[2]
    K1 = sum(p_z * kron(ket(i, 2), Z[i], sum(ZB)) for i = 1:2)
    K2 = sum((1 - p_z) * kron(ket(i, 2), X[i], sum(XB)) for i = 1:2)
    K = [K1, K2]
    Gρ = sum(K[i] * ρ * K[i]' for i = 1:2)
    return Gρ
end

function zmap(ρ)
    K = zkraus()
    Zρ = sum(K[i] * ρ * K[i] for i = 1:2)
    return Zρ
end

function zkraus()
    Z = [proj(1, 2), proj(2, 2)]
    K = [kron(Z[i], I(2)) for i = 1:2]
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

rate_bb84(::Type{T}, qz, qx) where {T} = hae_bb84(T, qz, qx) - hab_bb84(qz, qx)
rate_bb84(qz, qx) = rate_bb84(Float64, qz, qx)

hae_bb84_analytic(qz, qx) = 1 - binary_entropy(qx)
hab_bb84(qz, qx) = binary_entropy(qz)
rate_bb84_analytic(qz, qx) = hae_bb84_analytic(qz, qx) - hab_bb84(qz, qx)

function hae_bb84(::Type{T}, qz::Real, qx::Real) where {T}
    if qx == 0 && qz == 0
        return real(T(1))
    elseif qz == 0
        return hae_bb84_reducedz(T, qx)
    elseif qx == 0
        return hae_bb84_reducedx(T, qz)
    else
        return hae_bb84_general(T, qz, qx)
    end
end
hae_bb84(qz::Real, qx::Real) = hae_bb84(ComplexF64, qz, qx)

function hae_bb84_general(::Type{T}, qz, qx) where {T}
    R = real(T)
    is_complex = T <: Complex

    qz, qx = R.((qz, qx))
    model = GenericModel{R}()
    if is_complex
        @variable(model, ρ[1:4, 1:4], Hermitian)
    else
        @variable(model, ρ[1:4, 1:4], Symmetric)
    end
    corr_rho = corr(ρ)
    @constraint(model, corr_rho .== [qz, qx])
    @constraint(model, tr(ρ) == 1)

    Ghat = [I(4)]
    Zhat = zkraus()

    side = size(ρ, 1)
    vec_dim = Cones.svec_length(T, side)
    ρ_vec = Vector{GenericAffExpr{R,GenericVariableRef{R}}}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(R(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{R,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{R})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryz(::Type{T}) where {T} = [[1, 0, 0, 1] [1, 0, 0, -1]] / sqrt(T(2))
#isometryz2() = [[1,0,0,0] [0,0,0,1]]

function hae_bb84_reducedz(::Type{T}, qx) where {T}
    R = real(T)
    is_complex = T <: Complex

    qx = R(qx)
    model = GenericModel{R}()
    if is_complex
        @variable(model, ρ[1:2, 1:2], Hermitian)
    else
        @variable(model, ρ[1:2, 1:2], Symmetric)
    end
    @constraint(model, ρ[2, 2] == qx) #V'*QX*V == proj(2,2)
    @constraint(model, tr(ρ) == 1)

    V = isometryz(T)
    Ghat = [I(2)]
    Z = zkraus()
    V2 = [[1, 0, 0, 0] [0, 0, 0, 1]]
    Zhat = [V2'Zi * V for Zi in Z]

    side = size(ρ, 1)
    vec_dim = Cones.svec_length(T, side)
    ρ_vec = Vector{GenericAffExpr{R,GenericVariableRef{R}}}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(R(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{R,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{R})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryx(::Type{T}) where {T} = [[1, 0, 0, 1] [0, 1, 1, 0]] / sqrt(T(2))
#isometryx2() = 0.5 * [[1,1,1,1] [1,-1,-1,1]]

function hae_bb84_reducedx(::Type{T}, qz) where {T}
    R = real(T)
    is_complex = T <: Complex

    qz = R(qz)
    model = GenericModel{R}()
    if is_complex
        @variable(model, ρ[1:2, 1:2], Hermitian)
    else
        @variable(model, ρ[1:2, 1:2], Symmetric)
    end

    @constraint(model, ρ[2, 2] == qz) #V'*QZ*V == proj(2,2)
    @constraint(model, tr(ρ) == 1)

    V = isometryx(T)
    Ghat = [I(2)]
    Z = zkraus()
    Zhat = [Zi * V for Zi in Z]

    side = size(ρ, 1)
    vec_dim = Cones.svec_length(T, side)
    ρ_vec = Vector{GenericAffExpr{R,GenericVariableRef{R}}}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(R(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{R,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{R})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryxz(::Type{T}) where {T} = [1, 0, 0, 1] / sqrt(T(2))

function hae_bb84_reducedxz(::Type{T}) where {T}
    R = real(T)
    is_complex = T <: Complex

    model = GenericModel{R}()
    ρ = fill(T(1), (1, 1))

    V = isometryxz()
    Ghat = [fill(T(1), (1, 1))]
    Z = zkraus()
    V2 = [[1, 0, 0, 0] [0, 0, 0, 1]]
    Zhat = [V2'Zi * V for Zi in Z]

    side = size(ρ, 1)
    vec_dim = Cones.svec_length(R, side)
    ρ_vec = Vector{R}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1) * ρ, sqrt(R(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(R(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{R,T}(Ghat, Zhat, 1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{R})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end
