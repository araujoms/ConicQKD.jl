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

function zmap(ρ; T = Float64)
    K = zkraus(;T)
    Zρ = sum(K[i] * ρ * K[i] for i = 1:2)
    return Zρ
end

function zkraus(;T = Float64)
    Z = [proj(1, 2; T), proj(2, 2; T)]    
    K = [kron(Z[i], I(2)) for i = 1:2]
    return K
end

function corr(ρ; T = Float64)
    Z = [proj(1, 2; T), proj(2, 2; T)]
    X = [Complex{T}(0.5) * [1 1; 1 1], Complex{T}(0.5) * [1 -1; -1 1]]
    QZ = kron(Z[1],Z[2]) + kron(Z[2],Z[1])
    QX = kron(X[1],X[2]) + kron(X[2],X[1])
    global_basis = Hermitian.([QZ, QX])
    return real(dot.(Ref(ρ), global_basis))
end

total_rate_bb84(qz, qx; T = Float64) = hae_bb84(qz,qx; T) - hab_bb84(qz,qx; T)

hae_bb84_analytic(qz,qx) = (1 - binary_entropy(qx))
hab_bb84(qz,qx;T) = (1 - binary_entropy(qz))
total_rate_bb84_analytic(qz, qx; T = Float64) = hae_bb84_analytic(qz,qx;T) - hab_bb84(qz,qx;T)

function hae_bb84(qz,qx; T = Float64)
    if qx == 0 && qz == 0
        return T(1)
    elseif qz == 0
        return hae_bb84_reducedz(qx; T = Float64)
    elseif qx == 0
        return hae_bb84_reducedx(qz; T = Float64)
    else
        return hae_bb84_general(qz,qx; T = Float64)
    end
end

function hae_bb84_general(qz,qx; T = Float64)
    #R = Complex{T}
    R = T
    is_complex = R <: Complex
    
    qz, qx = T.((qz, qx))
    model = GenericModel{T}()
    if is_complex
        @variable(model, ρ[1:4, 1:4], Hermitian)
    else
        @variable(model, ρ[1:4, 1:4], Symmetric)
    end
    corr_rho = corr(ρ; T)
    @constraint(model, corr_rho .== [qz, qx])
    @constraint(model, tr(ρ) == T(1))

    Ghat = [I(4)]
    Zhat = zkraus(;T)

    side = size(ρ,1)
    vec_dim = Cones.svec_length(R, side)
    ρ_vec = Vector{GenericAffExpr{T,GenericVariableRef{T}}}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat,Zhat,1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryz(; T=Float64) = [[1,0,0,1] [1,0,0,-1]]/sqrt(T(2))
#isometryz2(; T=Float64) = T(1)*[[1,0,0,0] [0,0,0,1]]

function hae_bb84_reducedz(qx; T = Float64)
    #R = Complex{T}
    R = T
    is_complex = R <: Complex
    
    qx = T(qx)
    model = GenericModel{T}()
    if is_complex
        @variable(model, ρ[1:2, 1:2], Hermitian)
    else
        @variable(model, ρ[1:2, 1:2], Symmetric)
    end
    @constraint(model, ρ[2,2] == qx) #V'*QX*V == proj(2,2)
    @constraint(model, tr(ρ) == T(1))

    V = isometryz(;T)
    Ghat = [I(2)]
    Z = zkraus(;T)
    V2 = T(1)*[[1,0,0,0] [0,0,0,1]]
    Zhat = [V2'Zi*V for Zi in Z]

    side = size(ρ,1)
    vec_dim = Cones.svec_length(R, side)
    ρ_vec = Vector{GenericAffExpr{T,GenericVariableRef{T}}}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat,Zhat,1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryx(; T=Float64) = [[1,0,0,1] [0,1,1,0]]/sqrt(T(2))
#isometryx2(; T=Float64) = T(0.5)*[[1,1,1,1] [1,-1,-1,1]]

function hae_bb84_reducedx(qz; T = Float64)
    #R = Complex{T}
    R = T
    is_complex = R <: Complex
    
    qz = T(qz)
    model = GenericModel{T}()
    if is_complex
        @variable(model, ρ[1:2, 1:2], Hermitian)
    else
        @variable(model, ρ[1:2, 1:2], Symmetric)
    end
    
    @constraint(model, ρ[2,2] == qz) #V'*QZ*V == proj(2,2)
    @constraint(model, tr(ρ) == T(1))

    V = isometryx(;T)
    Ghat = [I(2)]
    Z = zkraus(;T)
    Zhat = [Zi*V for Zi in Z]

    side = size(ρ,1)
    vec_dim = Cones.svec_length(R, side)
    ρ_vec = Vector{GenericAffExpr{T,GenericVariableRef{T}}}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat,Zhat,1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end

isometryxz(; T=Float64) = [1,0,0,1]/sqrt(T(2))

function hae_bb84_reducedxz(; T = Float64)
    #R = Complex{T}
    R = T
    is_complex = R <: Complex
    
    model = GenericModel{T}()
    ρ = fill(R(1), (1,1))
    
    V = isometryxz(;T)
    Ghat = [fill(R(1), (1,1))]
    Z = zkraus(;T)
    V2 = T(1)*[[1,0,0,0] [0,0,0,1]]
    Zhat = [V2'Zi*V for Zi in Z]

    side = size(ρ,1)
    vec_dim = Cones.svec_length(R, side)
    ρ_vec = Vector{T}(undef, vec_dim)
    if is_complex
        Cones._smat_to_svec_complex!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    else
        Cones.smat_to_svec!(ρ_vec, T(1)*ρ, sqrt(T(2)))
    end

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat,Zhat,1 + vec_dim))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    return objective_value(model)
end
