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
using SparseArrays
import ForwardDiff
import GenericLinearAlgebra.eigen # needed by ForwardDiff currently for test_barrier
import Hypatia
import Hypatia.PolyUtils
import Hypatia.Cones
import Hypatia.RealOrComplex
import ConicQKD.EpiQKDTri

import ConicQKD.kraus2matrix
import ConicQKD.skron
import ConicQKD.svec
import ConicQKD.smat
import ConicQKD.symm_kron_full!

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
    prod_vec = zero(point)
    hess = Cones.hess(cone)
    @test hess * point ≈ dual_point atol = tol rtol = tol
    @test Cones.hess_prod!(prod_vec, point, cone) ≈ dual_point atol = tol rtol = tol
    inv_hess = Cones.inv_hess(cone)
    @test inv_hess * dual_point ≈ point atol = tol rtol = tol
    @test Cones.inv_hess_prod!(prod_vec, dual_point, cone) ≈ point atol = tol rtol = tol
    @test hess * inv_hess ≈ I atol = tol rtol = tol

    # generate random valid point
    R = eltype(cone.rho)
    rho = random_state(R, cone.d)
    Grho = Hermitian(smat(cone.G * svec(rho), R))
    Zrho = Hermitian(smat(cone.Z * svec(rho), R))
    relative_entropy = -von_neumann_entropy(Grho) + von_neumann_entropy(Zrho)
    point[1] = 2 * relative_entropy
    point[2:end] .= svec(rho)

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
    @test Cones.hess_prod!(prod_vec, point, cone) ≈ -grad atol = tol rtol = tol
    @test Cones.inv_hess_prod!(prod_vec, grad, cone) ≈ -point atol = tol rtol = tol

    prod_mat = zeros(T, dim, dim)
    @test Cones.hess_prod!(prod_mat, inv_hess, cone) ≈ I atol = tol rtol = tol
    @test Cones.inv_hess_prod!(prod_mat, hess, cone) ≈ I atol = tol rtol = tol

    psi = dual_point + grad
    proxsqr = dot(psi, Cones.inv_hess_prod!(prod_vec, psi, cone))
    @test Cones.get_proxsqr(cone, one(T), false) ≈ proxsqr atol = tol rtol = tol

    if hasproperty(cone, :use_hess_prod_slow)
        Cones.update_use_hess_prod_slow(cone)
        @test cone.use_hess_prod_slow_updated
        @test !cone.use_hess_prod_slow
        cone.use_hess_prod_slow = true
        @test Cones.hess_prod_slow!(prod_mat, inv_hess, cone) ≈ I atol = tol rtol = tol
    end

    if Cones.use_sqrt_hess_oracles(dim + 1, cone)
        prod_mat2 = Matrix(Cones.sqrt_hess_prod!(prod_mat, inv_hess, cone)')
        @test Cones.sqrt_hess_prod!(prod_mat, prod_mat2, cone) ≈ I atol = tol rtol = tol
        Cones.inv_sqrt_hess_prod!(prod_mat2, Matrix(one(T) * I, dim, dim), cone)
        @test prod_mat2' * prod_mat2 ≈ inv_hess atol = tol rtol = tol
    end

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
    # generate random valid point
    R = eltype(cone.rho)
    rho = random_state(R, cone.d)
    Grho = Hermitian(smat(cone.G * svec(rho), R))
    Zrho = Hermitian(smat(cone.Z * svec(rho), R))
    relative_entropy = -von_neumann_entropy(Grho) + von_neumann_entropy(Zrho)
    point[1] = 2 * relative_entropy
    point[2:end] .= svec(rho)

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
    @test Cones.inv_hess_prod!(prod_vec, fd_hess_dir, cone) ≈ dir atol = tol rtol = tol

    if Cones.use_dder3(cone)
        fd_third_dir = ForwardDiff.gradient(
            s2 -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> barrier_dir(s2, t), s), 0),
            TFD_point
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
    println("hess (with allocate)")
    @time Cones.hess(cone)
    println("inv_hess (with allocate)")
    @time Cones.inv_hess(cone)

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

    if Cones.use_sqrt_hess_oracles(dim + 1, cone)
        println("sqrt_hess_prod")
        @time Cones.sqrt_hess_prod!(point2, point1, cone)
        println("inv_sqrt_hess_prod")
        @time Cones.inv_sqrt_hess_prod!(point2, point1, cone)
    end

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

new_vec(w::Vector, dw::Int, T::Type{<:Real}) = copy(w)

function new_vec(w::Vector, dw::Int, R::Type{Complex{T}}) where {T<:Real}
    wR = zeros(Complex{eltype(w)}, dw)
    Cones.vec_copyto!(wR, w)
    return wR
end

function new_herm(w::Vector, dW::Int, T::Type{<:Real})
    W = similar(w, dW, dW)
    Cones.svec_to_smat!(W, w, sqrt(T(2)))
    return Hermitian(W, :U)
end

function new_herm(w::Vector, dW::Int, R::Type{Complex{T}}) where {T<:Real}
    W = zeros(Complex{eltype(w)}, dW, dW)
    Cones.svec_to_smat!(W, w, sqrt(T(2)))
    return Hermitian(W, :U)
end

function rand_sppsd_pattern(dW::Int)
    sparsity = inv(sqrt(dW))
    (row_idxs, col_idxs, _) = findnz(tril!(sprand(Bool, dW, dW, sparsity)) + I)
    return (row_idxs, col_idxs)
end

function rand_herms(ds::Int, Rd::Vector, T::Type{<:Real})
    Ps = Vector{LinearAlgebra.HermOrSym{R,Matrix{R}} where {R<:RealOrComplex{T}}}(undef, length(Rd))
    A_1_half = randn(Rd[1], ds, ds)
    Ps[1] = Hermitian(A_1_half * A_1_half' + I, :U)
    for i = 2:length(Rd)
        Ps[i] = Hermitian(randn(Rd[i], ds, ds), :U)
    end
    return Ps
end

function rand_powers(T, d)
    Random.seed!(1)
    α = rand(T, d) .+ 1
    α ./= sum(α)
    return α
end

# real Ps for WSOS cones, use unit box domain
function rand_interp(num_vars::Int, halfdeg::Int, T::Type{<:Real})
    Random.seed!(1)
    domain = PolyUtils.BoxDomain{T}(-ones(T, num_vars), ones(T, num_vars))
    (d, _, Ps, _) = PolyUtils.interpolate(domain, halfdeg; sample = false)
    return (d, Ps)
end

# complex Ps for WSOS cones, use unit ball domain
function rand_interp(num_vars::Int, halfdeg::Int, R::Type{<:Complex{<:Real}})
    Random.seed!(1)
    gs = [z -> 1 - sum(abs2, z)]
    g_halfdegs = [1]
    (points, Ps) = PolyUtils.interpolate(R, halfdeg, num_vars, gs, g_halfdegs)
    d = length(points)
    return (d, Ps)
end

# cones

# EpiQKDTri
function von_neumann_entropy(rho)
    λ = eigvals(rho)
    return -dot(λ, log.(λ))
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
    Z = [kron(proj(R, i, dout) * V, I(din)) for i = 1:dout]

    return G, Z, rho_dim, rho_idxs
end

function test_oracles(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    G, Z, rho_dim, rho_idxs = random_protocol(din, dout, R)
    test_oracles(cone(G, Z, 1 + rho_dim); init_tol = Inf)
end

function test_barrier(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 3, 4
    gkraus, zkraus, rho_dim, rho_idxs = random_protocol(din, dout, R)
    G = kraus2matrix(gkraus, R)
    Z = kraus2matrix(zkraus, R)

    function barrier(point)
        u = point[1]
        rhoH = new_herm(point[rho_idxs], din^2, R)
        GrhoH = new_herm(G * point[rho_idxs], din^2, R)
        ZrhoH = new_herm(Z * point[rho_idxs], dout * din, R)
        relative_entropy = -von_neumann_entropy(GrhoH) + von_neumann_entropy(ZrhoH)
        return -real(log(u - relative_entropy)) - logdet_pd(rhoH)
    end
    return test_barrier(cone(gkraus, zkraus, 1 + rho_dim), barrier; TFD = Float64)
end

function show_time_alloc(cone::Type{EpiQKDTri{T,R}}) where {T,R}
    din, dout = 4, 5
    G, Z, rho_dim, rho_idxs = random_protocol(din, dout, R)
    return show_time_alloc(cone(G, Z, 1 + rho_dim))
end
