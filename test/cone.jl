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
import Random.randn
using LinearAlgebra
import BlockDiagonals
import ForwardDiff
import GenericLinearAlgebra.eigen # needed by ForwardDiff currently for test_barrier
import Hypatia
import Hypatia.PolyUtils
import Hypatia.Cones
import Hypatia.RealOrComplex
import Hypatia.Cones.svec_to_smat!

import ConicQKD.EpiQKDTri
import ConicQKD.EpiRenyiQKDTri
import ConicQKD.d3Ψdρ3!
import ConicQKD.kraus2matrix
import ConicQKD.skron
import ConicQKD.svec
import ConicQKD.smat

Random.randn(::Type{BigFloat}, dims::Integer...) = BigFloat.(randn(dims...))

function Random.randn(::Type{Complex{BigFloat}}, dims::Integer...)
    return Complex{BigFloat}.(randn(ComplexF64, dims...))
end

function random_state(::Type{T}, d::Integer, k::Integer = d) where {T}
    Random.seed!(1)
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
    tol::Real = 1e8 * eps(T),
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
    # prod_vec = zero(point)
    # hess = Cones.hess(cone)
    # @test hess * point ≈ dual_point atol = tol rtol = tol
    # @test Cones.hess_prod!(prod_vec, point, cone) ≈ dual_point atol = tol rtol = tol
    # inv_hess = Cones.inv_hess(cone)
    # @test inv_hess * dual_point ≈ point atol = tol rtol = tol
    # @test Cones.inv_hess_prod!(prod_vec, dual_point, cone) ≈ point atol = tol rtol = tol
    # @test hess * inv_hess ≈ I atol = tol rtol = tol

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
        @test -Cones.dder3(cone, point) ≈ grad atol = tol rtol = tol

        dir = perturb_scale!(zeros(T, dim), noise, one(T))
        dder3 = Cones.dder3(cone, dir)
        @test dot(dder3, point) ≈ dot(dir, hess * dir) atol = tol rtol = tol
    end

    return
end

# check some oracles agree with ForwardDiff
function test_barrier(
    cone::Cones.Cone{T},
    barrier::Function;
    noise::T = T(1e-1),
    scale::T = T(1e-1),
    tol::Real = 1e8 * eps(T),
    TFD::Type{<:Real} = T
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
    TFD_point = TFD.(point)

    fd_grad = ForwardDiff.gradient(barrier, TFD_point)
    @test Cones.grad(cone) ≈ fd_grad atol = tol rtol = tol

    dir = 10 * randn(T, dim)
    TFD_dir = TFD.(dir)

    barrier_dir(s, t) = barrier(s + t * TFD_dir)

    fd_hess_dir = ForwardDiff.gradient(s -> ForwardDiff.derivative(t -> barrier_dir(s, t), 0), TFD_point)

    @test Cones.hess(cone) * dir ≈ fd_hess_dir atol = tol rtol = tol
    @test Cones.inv_hess(cone) * fd_hess_dir ≈ dir atol = tol rtol = tol
    prod_vec = zero(dir)
    @test Cones.hess_prod!(prod_vec, dir, cone) ≈ fd_hess_dir atol = tol rtol = tol

    if Cones.use_dder3(cone)
        fd_third_dir = ForwardDiff.gradient(
            s2 -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> barrier_dir(s2, t), s), 0),
            TFD_point
        )

        @test -2 * Cones.dder3(cone, dir) ≈ fd_third_dir atol = tol rtol = tol
    end

    return
end

# Test d3Ψdρ3
function test_d3Ψdρ3(
    cone::Cones.Cone{T},
    renyi2_barrier::Function;
    noise::T = T(1e-1),
    scale::T = T(1e-1),
    tol::Real = 1e8 * eps(T),
    TFD::Type{<:Real} = T
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
    TFD_point = TFD.(point)

    dir = 10 * randn(T, dim - 1)
    TFD_dir = TFD.(dir)

    renyi2_dir(s, t) = renyi2_barrier(s + t * TFD_dir)

    # grad = ForwardDiff.derivative(t -> renyi2_dir(TFD_point[2:dim], t), 0)
    # fd_hess_dir = ForwardDiff.gradient(s -> ForwardDiff.derivative(t -> renyi2_dir(s, t), 0), TFD_point[2:dim])

    if Cones.use_dder3(cone)
        
        fd_third_dir = ForwardDiff.gradient(
            s2 -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> renyi2_dir(s2, t), s), 0),
            TFD_point[2:dim]
        )

        ρ_dir_mat = cone.mat
        svec_to_smat!(ρ_dir_mat, dir, cone.rt2)

        d3Ψdρ3!(cone.d2Ψdρ2vec, ρ_dir_mat, cone)

        @test cone.d2Ψdρ2vec ≈ fd_third_dir atol = tol rtol = tol
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

function renyi1(ρ::Hermitian, σ::Hermitian, α::T, S) where {T<:Real}
    αexp = (1 - α) / (2 * α)
    λ, U = eigen(σ)
    λ = map(x -> x <= 0 ? T(0) : x^αexp, λ)
    σpower = U * Diagonal(λ) * U' * S
    λ = eigvals(Hermitian(σpower * ρ * σpower'))
    λ = max.(T(0), λ)
    return sum(λ .^ α)
end

function renyi2(ρ, σ, α, S)
    αexp = (1 - α) / 2α
    ZG = σ^αexp * S * sqrt(ρ)
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

function random_protocol(din::Integer, dout::Integer, R::Type)
    rho_dim = Cones.svec_length(R, din^2)
    rho_idxs = 2:(rho_dim+1)

    U = random_unitary(R, dout)
    V = U[:, 1:din]

    G = [random_unitary(R, din^2)]
    #    G = [R.(I(din^2))]
    Z = [kron(proj(R, i, dout) * V, I(din)) for i ∈ 1:dout]

    blocks = [(i-1)*din+1:i*din for i ∈ 1:dout]

    return G, Z, rho_dim, rho_idxs, blocks
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
    G, Z, rho_dim, rho_idxs, blocks = random_protocol(din, dout, R)
    test_oracles(cone(G, Z, 1 + rho_dim; blocks); init_tol = Inf)
end

function test_barrier(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    gkraus, zkraus, rho_dim, rho_idxs, blocks = random_protocol(din, dout, R)
    G = kraus2matrix(gkraus)
    Z = kraus2matrix(zkraus)

    function barrier(point)
        u = point[1]
        rhoH = smat(point[rho_idxs])
        GrhoH = smat(G * point[rho_idxs])
        ZrhoH = smat(Z * point[rho_idxs])
        relative_entropy = -von_neumann_entropy(GrhoH) + von_neumann_entropy(ZrhoH)
        return -real(log(u - relative_entropy)) - logdet_pd(rhoH)
    end
    return test_barrier(cone(gkraus, zkraus, 1 + rho_dim; blocks), barrier; TFD = Float64)
end

function show_time_alloc(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 4, 5
    G, Z, rho_dim, rho_idxs, blocks = random_protocol(din, dout, R)
    return show_time_alloc(cone(G, Z, 1 + rho_dim; blocks))
end

function random_point!(point, cone::EpiRenyiQKDTri{T,R}) where {T,R}
    rho = random_state(R, cone.d)
    Grho = smat(cone.G * svec(rho))
    S = cone.S
    Zrhoblocks = smat.(cone.Z .* Ref(svec(rho)))
    Zrho = Hermitian(Matrix(BlockDiagonals.BlockDiagonal(Zrhoblocks)))
    r = renyi2(Grho, Zrho, cone.α, S)
    point[1] = cone.sα * r + 0.1
    point[2:end] .= svec(rho)
end

function test_oracles(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    α, G, Z, rho_dim, rho_idxs, blocks, S = random_protocol(cone, din, dout)
    test_oracles(cone(α, G, Z, 1 + rho_dim; S, blocks); init_tol = Inf)
end

function random_protocol(cone::Type{EpiRenyiQKDTri{T,R}}, din::Integer, dout::Integer) where {T,R}
    α = T(9) / 10

    rho_dim = Cones.svec_length(R, din^2)
    rho_idxs = 2:(rho_dim+1)

    U = random_unitary(R, dout)
    V = U[:, 1:din]

    G = [random_unitary(R, din^2)]
    #G = [R.(I(din^2))]
    Z = [kron(proj(R, i, dout) * V, I(din)) for i ∈ 1:dout]

    blocks = [(i-1)*din+1:i*din for i ∈ 1:dout]

    return α, G, Z, rho_dim, rho_idxs, blocks, kron(V, I(din))
end

function test_barrier(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 2, 3
    α, gkraus, zkraus, rho_dim, rho_idxs, blocks, S = random_protocol(cone, din, dout)
    sα = α < 1 ? -1 : 1
    G = kraus2matrix(gkraus)
    Z = kraus2matrix(zkraus)

    function barrier(point)
        u = point[1]
        rhoH = smat(point[rho_idxs])
        GrhoH = smat(G * point[rho_idxs])
        ZrhoH = smat(Z * point[rho_idxs])
        r = renyi2(GrhoH, ZrhoH, α, S)
        return -real(log(u - sα * r)) - logdet_pd(rhoH)
    end
    return test_barrier(cone(α, gkraus, zkraus, 1 + rho_dim; S, blocks), barrier; TFD = Float64)
end


function test_d3Ψdρ3(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 2, 3
    α, gkraus, zkraus, rho_dim, rho_idxs, blocks, S = random_protocol(cone, din, dout)
    G = kraus2matrix(gkraus)
    Z = kraus2matrix(zkraus)

    function renyi2_barrier(point)
        GrhoH = smat(G * point)
        ZrhoH = smat(Z * point)
        return renyi2(GrhoH, ZrhoH, α, S)
    end
    return test_d3Ψdρ3(cone(α, gkraus, zkraus, 1 + rho_dim; S, blocks), renyi2_barrier; TFD = Float64)
end


function show_time_alloc(cone::Type{EpiRenyiQKDTri{T,R}}) where {T,R}
    din, dout = 4, 5
    G, Z, rho_dim, rho_idxs, blocks = random_protocol(din, dout, R)
    return show_time_alloc(cone(G, Z, 1 + rho_dim; blocks))
end
