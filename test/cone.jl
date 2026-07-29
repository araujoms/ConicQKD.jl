#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
tests for primitive cone barrier oracles
=#

using Test
import Random
using LinearAlgebra
import ForwardDiff
import GenericLinearAlgebra.eigen # needed by ForwardDiff currently for test_barrier
import Hypatia
import Hypatia.PolyUtils
import Hypatia.Cones
import Hypatia.RealOrComplex
import ConicQKD: EpiQKDTri, EpiRenyiQKDTri, EpiFastRenyiQKDTri
import ConicQKD: EpiQKDTriCone, EpiRenyiQKDTriCone, EpiFastRenyiQKDTriCone
import ConicQKD: kraus2matrix, skron, svec, smat
import JuMP
import Dualization

function random_state(::Type{T}, d::Integer, k::Integer = d) where {T}
    #Random.seed!(1)
    x = randn(T, (d, k))
    y = x * x'
    y ./= tr(y)
    return Hermitian(y)
end

# sanity check oracles
function test_oracles(
    cone::Cones.Cone{T};
    noise::T = T(1e-1),
    scale::T = T(1e-1),
    tol::Real = sqrt(eps(T)),
    init_only::Bool = false,
    init_tol::Real = tol
) where {T<:Real}
    Random.seed!(1)
    dim = Cones.dimension(cone)
    Cones.setup_data!(cone)
    Cones.reset_data(cone)

    point = zeros(T, dim)
    Cones.set_initial_point!(point, cone)
    Cones.load_point(cone, point)
    @test Cones.is_feas(cone)
    @test cone.point == point

    dual_point = -Cones.grad(cone)
    Cones.load_dual_point(cone, dual_point)
    @test Cones.is_dual_feas(cone)
    @test cone.dual_point == dual_point
    @test Cones.get_proxsqr(cone, one(T), true) <= 1 # max proximity
    @test Cones.get_proxsqr(cone, one(T), false) <= dim # sum proximity

    # test centrality of initial point
    if isfinite(init_tol)
        @test point ≈ dual_point atol = init_tol rtol = init_tol
    end
    init_only && return

    # test at initial point
    prod_vec = zero(point)
    hess = Cones.hess(cone)
    @test hess * point ≈ dual_point atol = tol rtol = tol
    @test Cones.hess_prod!(prod_vec, point, cone) ≈ dual_point atol = tol rtol = tol

    # generate random valid point
    random_point!(point, cone)

    Cones.reset_data(cone)
    Cones.load_point(cone, point)
    @test Cones.is_feas(cone)
    dual_point = -Cones.grad(cone)
    Cones.load_dual_point(cone, dual_point)
    @test Cones.is_dual_feas(cone)

    # test gradient and Hessian oracles
    nu = Cones.get_nu(cone)
    grad = Cones.grad(cone)
    @test dot(point, grad) ≈ -nu atol = tol rtol = tol

    hess = Matrix(Cones.hess(cone))
    inv_hess = Matrix(Cones.inv_hess(cone))
    @test hess * inv_hess ≈ I atol = tol rtol = tol

    @test hess * point ≈ -grad atol = tol rtol = tol
    prod_vec = zero(point)

    @test Cones.hess_prod!(prod_vec, point, cone) ≈ -grad atol = tol rtol = tol

    prod_mat = zeros(T, dim, dim)
    @test Cones.hess_prod!(prod_mat, inv_hess, cone) ≈ I atol = tol rtol = tol

    psi = dual_point + grad
    proxsqr = dot(psi, Cones.inv_hess_prod!(prod_vec, psi, cone))
    @test Cones.get_proxsqr(cone, one(T), false) ≈ proxsqr atol = tol rtol = tol

    # test third order deriv oracle
    if Cones.use_dder3(cone)
        @test -Cones.dder3(cone, point) ≈ grad atol = sqrt(tol) rtol = sqrt(tol)

        dir = perturb_scale!(zeros(T, dim), noise, one(T))
        dder3 = Cones.dder3(cone, dir)
        @test dot(dder3, point) ≈ dot(dir, hess * dir) atol = sqrt(tol) rtol = sqrt(tol)
    end

    return
end

# check some oracles agree with ForwardDiff
function test_barrier(
    cone::Cones.Cone{T},
    barrier::Function;
    noise::T = T(1e-1),
    scale::T = T(1e-1),
    tol::Real = sqrt(eps(T))
) where {T<:Real}
    Random.seed!(1)
    dim = Cones.dimension(cone)
    Cones.setup_data!(cone)

    point = zeros(T, dim)
    Cones.set_initial_point!(point, cone)
    random_point!(point, cone)

    Cones.reset_data(cone)
    Cones.load_point(cone, point)
    @test Cones.is_feas(cone)

    fd_grad = ForwardDiff.gradient(barrier, point)
    @test Cones.grad(cone) ≈ fd_grad atol = tol rtol = tol

    dir = 10 * randn(T, dim)
    barrier_dir(s, t) = barrier(s + t * dir)

    fd_hess_dir = ForwardDiff.gradient(s -> ForwardDiff.derivative(t -> barrier_dir(s, t), 0), point)

    @test Cones.hess(cone) * dir ≈ fd_hess_dir atol = tol rtol = tol
    @test Cones.inv_hess(cone) * fd_hess_dir ≈ dir atol = tol rtol = tol
    prod_vec = zero(dir)
    @test Cones.hess_prod!(prod_vec, dir, cone) ≈ fd_hess_dir atol = tol rtol = tol

    if Cones.use_dder3(cone)
        fd_third_dir = ForwardDiff.gradient(
            s2 -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> barrier_dir(s2, t), s), 0),
            point
        )

        @test -2 * Cones.dder3(cone, dir) ≈ fd_third_dir atol = tol rtol = tol
    end

    return
end

# show time and memory allocation for oracles
function show_time_alloc(cone::Cones.Cone{T}; noise::T = T(1e-4), scale::T = T(1e-1)) where {T<:Real}
    Random.seed!(1)
    dim = Cones.dimension(cone)
    println("dimension: ", dim)

    println("setup_data")
    @time Cones.setup_data!(cone)
    Cones.reset_data(cone)

    point = zeros(T, dim)
    Cones.set_initial_point!(point, cone)
    perturb_scale!(point, noise, scale)
    Cones.load_point(cone, point)
    @assert Cones.is_feas(cone)

    dual_point = -Cones.grad(cone)
    perturb_scale!(dual_point, noise, inv(scale))
    Cones.load_dual_point(cone, dual_point)
    @assert Cones.is_dual_feas(cone)

    Cones.reset_data(cone)

    Cones.load_point(cone, point)
    println("is_feas")
    @time Cones.is_feas(cone)

    Cones.load_dual_point(cone, dual_point)
    println("is_dual_feas")
    @time Cones.is_dual_feas(cone)

    println("grad")
    @time Cones.grad(cone)
    #    println("hess (with allocate)")
    #    @time Cones.hess(cone)
    #    println("inv_hess (with allocate)")
    #    @time Cones.inv_hess(cone)

    point1 = randn(T, dim)
    point2 = zero(point1)
    println("hess_prod")
    @time Cones.hess_prod!(point2, point1, cone)
    println("inv_hess_prod")
    @time Cones.inv_hess_prod!(point2, point1, cone)

    if hasproperty(cone, :use_hess_prod_slow)
        cone.use_hess_prod_slow_updated = true
        cone.use_hess_prod_slow = true
        println("hess_prod_slow")
        @time Cones.hess_prod_slow!(point2, point1, cone)
    end

    #    if Cones.use_sqrt_hess_oracles(dim + 1, cone)
    #        println("sqrt_hess_prod")
    #        @time Cones.sqrt_hess_prod!(point2, point1, cone)
    #        println("inv_sqrt_hess_prod")
    #        @time Cones.inv_sqrt_hess_prod!(point2, point1, cone)
    #    end

    if Cones.use_dder3(cone)
        println("dder3")
        @time Cones.dder3(cone, point1)
    end

    println("get_proxsqr")
    @time Cones.get_proxsqr(cone, one(T), true)

    return
end

function perturb_scale!(point::Vector{T}, noise::T, scale::T) where {T<:Real}
    if !iszero(noise)
        @. point += 2 * noise * rand(T) - noise
    end
    if !isone(scale)
        point .*= scale
    end
    return point
end

# cone utilities

logdet_pd(W::Hermitian) = logdet(cholesky!(copy(W)))
logdet_pd(W::Symmetric) = logdet(cholesky!(copy(W)))

# EpiQKDTri
function von_neumann_entropy(rho)
    λ = eigvals(rho)
    return -dot(λ, log.(λ))
end

function renyi(ρ, σ, α, S)
    αexp = (1 - α) / 2α
    ZG = σ^αexp * S * sqrt(ρ)
    λ = svdvals(ZG)
    return sum(λ .^ 2α)
end

function renyi_blocks(ρ, σ::Vector{<:AbstractMatrix}, α, S)
    αexp = (1 - α) / 2α
    σpower = σ .^ αexp
    sizes = size.(σ, 1)
    Ssqrtρ = S * sqrt(ρ)
    ZG = zeros(eltype(ρ), sum(sizes), size(Ssqrtρ, 2))
    c = 0
    for σi ∈ σpower
        idxs = 1+c:size(σi, 1)+c
        ZG[idxs, :] .= σi * Ssqrtρ[idxs, :]
        c += size(σi, 1)
    end
    λ = svdvals(ZG)
    return sum(λ .^ 2α)
end

function proj(::Type{T}, i::Integer, d::Integer) where {T<:Number}
    p = Hermitian(zeros(T, d, d))
    p[i, i] = 1
    return p
end

function random_unitary(::Type{T}, d::Integer) where {T<:Number}
    z = randn(T, (d, d))
    Q, R = qr(z)
    Λ = sign.(real(Diagonal(R)))
    return Q * Λ
end

function random_point!(point, cone::EpiQKDTri{T,R}) where {T,R}
    rho = random_state(R, cone.d)
    Grho = smat(cone.G * svec(rho))
    Zrho = smat.(cone.Z .* Ref(svec(rho)))
    relative_entropy = -von_neumann_entropy(Grho) + sum(von_neumann_entropy.(Zrho))
    point[1] = 2 * relative_entropy
    point[2:end] .= svec(rho)
end

function test_oracles(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    _, G, Z, rho_dim, blocks, _ = random_protocol(cone, din, dout)
    test_oracles(cone(G, Z, 1 + rho_dim; blocks); init_tol = Inf)
end

function test_barrier(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    _, gkraus, zkraus, rho_dim, blocks, _ = random_protocol(cone, din, dout)
    G = kraus2matrix(gkraus)
    Z = kraus2matrix(zkraus)

    function barrier(point)
        u = point[1]
        rhoH = smat(point[2:end])
        GrhoH = smat(G * point[2:end])
        ZrhoH = smat(Z * point[2:end])
        relative_entropy = -von_neumann_entropy(GrhoH) + von_neumann_entropy(ZrhoH)
        return -real(log(u - relative_entropy)) - logdet_pd(rhoH)
    end
    return test_barrier(cone(gkraus, zkraus, 1 + rho_dim; blocks), barrier)
end

function show_time_alloc(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 4, 5
    _, G, Z, rho_dim, blocks, _ = random_protocol(cone, din, dout)
    return show_time_alloc(cone(G, Z, 1 + rho_dim; blocks))
end

function random_point!(point, cone::EpiRenyiQKDTri{T,R}) where {T,R}
    ρ = random_state(R, cone.ρd)
    σ = random_state(R, cone.σd)
    Gρ = smat(cone.G * svec(ρ))
    S = cone.S
    Zσ = smat.(cone.Z .* Ref(svec(σ)))
    r = renyi_blocks(Gρ, Zσ, cone.α, S)
    point[1] = cone.sα * r + 0.1
    point[cone.ρ_idxs] .= svec(ρ)
    point[cone.σ_idxs] .= svec(σ)
end

function random_point!(point, cone::EpiFastRenyiQKDTri{T,R}) where {T,R}
    rho = random_state(R, cone.d)
    Grho = smat(cone.G * svec(rho))
    S = cone.S
    Zrhoblocks = smat.(cone.Z .* Ref(svec(rho)))
    r = renyi_blocks(Grho, Zrhoblocks, cone.α, S)
    point[1] = cone.sα * r + 0.1
    point[2:end] .= svec(rho)
end

function test_oracles(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    α, G, Z, rho_dim, blocks, S = random_protocol(cone, din, dout)
    test_oracles(cone(α, G, Z, 1 + 2rho_dim; S, blocks); init_tol = Inf)
end

function test_oracles(cone::Type{<:EpiFastRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    α, G, Z, rho_dim, blocks, S = random_protocol(cone, din, dout)
    test_oracles(cone(α, G, Z, 1 + rho_dim; S, blocks); init_tol = Inf)
end

const EntropyCones{T,R} = Union{EpiQKDTri{T,R},EpiRenyiQKDTri{T,R},EpiFastRenyiQKDTri{T,R}}
function random_protocol(::Type{<:EntropyCones{T,R}}, din::Integer, dout::Integer) where {T,R}
    α = T(9) / 10

    rho_dim = Cones.svec_length(R, din^2)

    U = random_unitary(R, dout)
    V = U[:, 1:din]

    G = [random_unitary(R, din^2)]
    Z = [kron(proj(R, i, dout) * V, I(din)) for i ∈ 1:dout]

    blocks = [(i-1)*din+1:i*din for i ∈ 1:dout]

    return α, G, Z, rho_dim, blocks, kron(V, I(din))
end

function test_barrier(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 2, 3
    α, gkraus, zkraus, rho_dim, blocks, S = random_protocol(cone, din, dout)
    sα = α < 1 ? -1 : 1
    G = kraus2matrix(gkraus)
    Z = kraus2matrix(zkraus)

    function barrier(point)
        u = point[1]
        ρvec = point[2:rho_dim+1]
        σvec = point[rho_dim+2:end]
        ρ = smat(ρvec)
        σ = smat(σvec)
        Gρ = smat(G * ρvec)
        Zσ = smat(Z * σvec)
        r = renyi(Gρ, Zσ, α, S)
        return -real(log(u - sα * r)) - logdet_pd(ρ) - logdet_pd(σ)
    end
    return test_barrier(cone(α, gkraus, zkraus, 1 + 2rho_dim; S, blocks), barrier)
end

function test_barrier(cone::Type{EpiFastRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 2, 3
    α, gkraus, zkraus, rho_dim, blocks, S = random_protocol(cone, din, dout)
    sα = α < 1 ? -1 : 1
    G = kraus2matrix(gkraus)
    Z = kraus2matrix(zkraus)

    function barrier(point)
        u = point[1]
        ρvec = point[2:rho_dim+1]
        ρ = smat(ρvec)
        Gρ = smat(G * ρvec)
        Zρ = smat(Z * ρvec)
        r = renyi(Gρ, Zρ, α, S)
        return -real(log(u - sα * r)) - logdet_pd(ρ)
    end
    return test_barrier(cone(α, gkraus, zkraus, 1 + rho_dim; S, blocks), barrier)
end

function show_time_alloc(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    α, gkraus, zkraus, rho_dim, blocks, S = random_protocol(cone, din, dout)
    return show_time_alloc(cone(α, gkraus, zkraus, 1 + 2rho_dim; S, blocks))
end

function show_time_alloc(cone::Type{EpiFastRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    α, gkraus, zkraus, rho_dim, blocks, S = random_protocol(cone, din, dout)
    return show_time_alloc(cone(α, gkraus, zkraus, 1 + rho_dim; S, blocks))
end

function test_dual(conetype::Type{<:EntropyCones{T,R}}) where {T,R}
    din, dout = 2, 3
    α, gkraus, zkraus, rho_dim, blocks, S = random_protocol(conetype, din, dout)
    sα = α < 1 ? -1 : 1

    model = JuMP.GenericModel{T}()
    ρ = random_state(R, din^2)
    σ = random_state(R, din^2)
    JuMP.@variable(model, h)
    if conetype <: EpiQKDTri{T,R}
        JuMP.@constraint(model, [h; svec(ρ)] in EpiQKDTriCone{T,R}(gkraus, zkraus, 1 + rho_dim; blocks))
    elseif conetype <: EpiFastRenyiQKDTri{T,R}
        JuMP.@constraint(model, [h; svec(ρ)] in EpiFastRenyiQKDTriCone{T,R}(α, gkraus, zkraus, 1 + rho_dim; S, blocks))
    else
        JuMP.@constraint(model, [h; svec(ρ); svec(σ)] in EpiRenyiQKDTriCone{T,R}(α, gkraus, zkraus, 1 + 2rho_dim; S, blocks))
    end
    JuMP.@objective(model, Min, h)
    JuMP.set_optimizer(model, Hypatia.Optimizer{T})
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    primal_objective = JuMP.objective_value(model)
    JuMP.set_optimizer(model, Dualization.dual_optimizer(Hypatia.Optimizer{T}; coefficient_type = T))
    JuMP.optimize!(model)
    dual_objective = JuMP.objective_value(model)
    @test primal_objective ≈ dual_objective

    return
end
