import ForwardDiff

"""
$(TYPEDEF)

Epigraph of QKD cone of dimension `dim` in svec format.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiQKDTri{T<:Real,R<:RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    Gd::Int
    Zd::Int
    is_complex::Bool
    Gkraus::Vector
    Zkraus::Vector

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    hess_aux_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    dder3_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T,Matrix{T}}
    inv_hess::Symmetric{T,Matrix{T}}
    hess_fact_mat::Symmetric{T,Matrix{T}}
    hess_fact::Factorization{T}

    rt2::T
    is_G_identity::Bool
    rho_dim::Int
    Grho_dim::Int
    Zrho_dim::Int
    rho_idxs::UnitRange{Int}
    rho::Matrix{R}
    Grho::Matrix{R}
    Zrho::Matrix{R}
    Grho_log::Matrix{R}
    Zrho_log::Matrix{R}
    G::Matrix{T}
    Z::Matrix{T}
    Gadj::Matrix{T}
    Zadj::Matrix{T}
    rho_fact::Eigen{R}
    Grho_fact::Eigen{R}
    Zrho_fact::Eigen{R}
    rho_inv::Matrix{R}
    rho_λ_log::Vector{T}
    Grho_λ_log::Vector{T}
    Zrho_λ_log::Vector{T}
    z::T
    dzdrho::Vector{T}
    Δ2G::Matrix{T}
    Δ2Z::Matrix{T}
    d2zdrho2::Matrix{T}
    d2zdrho2G::Matrix{T}
    d2zdrho2Z::Matrix{T}
    #variables below are just scratch space
    mat::Matrix{R}
    Gmat::Matrix{R}
    Zmat::Matrix{R}
    Gmat2::Matrix{R}
    Zmat2::Matrix{R}
    Gmat3::Matrix{R}
    Zmat3::Matrix{R}
    big_Gmat::Matrix{T}
    big_Zmat::Matrix{T}
    vec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{T}

    function EpiQKDTri{T,R}(Gkraus::Vector, Zkraus::Vector, dim::Int; use_dual::Bool = false) where {T<:Real,R<:RealOrComplex{T}}
        @assert dim > 1
        cone = new{T,R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.rho_dim = dim - 1
        cone.d = svec_side(R, cone.rho_dim)
        cone.Gd = size(Gkraus[1], 1)
        cone.Zd = size(Zkraus[1], 1)
        cone.G = kraus2matrix(Gkraus, T, R)
        cone.Z = kraus2matrix(Zkraus, T, R)
        cone.Gadj = Matrix(cone.G')
        cone.Zadj = Matrix(cone.Z')
        cone.is_G_identity = cone.G == I(cone.rho_dim)
        cone.Grho_dim = size(cone.G, 1)
        cone.Zrho_dim = size(cone.Z, 1)
        return cone
    end
end

use_dder3(cone::EpiQKDTri) = false

function reset_data(cone::EpiQKDTri)
    return (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.hess_aux_updated = cone.inv_hess_updated = cone.hess_fact_updated = cone.dder3_aux_updated = false)
end

function setup_extra_data!(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    rho_dim = cone.rho_dim
    Grho_dim = cone.Grho_dim
    Zrho_dim = cone.Zrho_dim
    d = cone.d
    Gd = cone.Gd
    Zd = cone.Zd
    cone.rt2 = sqrt(T(2))
    cone.rho_idxs = 2:(rho_dim+1)
    cone.rho = zeros(R, d, d)
    cone.Grho = zeros(R, Gd, Gd)
    cone.Zrho = zeros(R, Zd, Zd)
    cone.Grho_log = zeros(R, Gd, Gd)
    cone.Zrho_log = zeros(R, Zd, Zd)
    cone.rho_inv = zeros(R, d, d)
    cone.dzdrho = zeros(T, rho_dim)
    cone.Δ2G = zeros(T, Gd, Gd)
    cone.Δ2Z = zeros(T, Zd, Zd)
    cone.d2zdrho2 = zeros(T, rho_dim, rho_dim)
    cone.d2zdrho2G = zeros(T, Grho_dim, Grho_dim)
    cone.d2zdrho2Z = zeros(T, Zrho_dim, Zrho_dim)
    cone.rho_λ_log = zeros(T, d)
    cone.Grho_λ_log = zeros(T, Gd)
    cone.Zrho_λ_log = zeros(T, Zd)

    cone.mat = zeros(R, d, d)
    cone.Gmat = zeros(R, Gd, Gd)
    cone.Zmat = zeros(R, Zd, Zd)
    cone.Gmat2 = zeros(R, Gd, Gd)
    cone.Zmat2 = zeros(R, Zd, Zd)
    cone.Gmat3 = zeros(R, Gd, Gd)
    cone.Zmat3 = zeros(R, Zd, Zd)
    cone.big_Gmat = zeros(T, rho_dim, Grho_dim)
    cone.big_Zmat = zeros(T, rho_dim, Zrho_dim)
    cone.vec = zeros(T, rho_dim)
    cone.Gvec = zeros(T, Grho_dim)
    cone.Zvec = zeros(T, Zrho_dim)
    return
end

get_nu(cone::EpiQKDTri) = cone.d + 1

function set_initial_point!(arr::AbstractVector{T}, cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    for i = 1:(cone.d)
        arr[1+k] = T(1)
        k += incr * i + 1
    end
    @views rho_vec = arr[cone.rho_idxs]
    svec_to_smat!(cone.Grho, cone.G * rho_vec, cone.rt2)
    svec_to_smat!(cone.Zrho, cone.Z * rho_vec, cone.rt2)
    Grho_λ = cone.Grho_λ_log = eigvals(Hermitian(cone.Grho))
    Zrho_λ = cone.Zrho_λ_log = eigvals(Hermitian(cone.Zrho))
    relative_entropy = dot(Grho_λ, log.(Grho_λ)) - dot(Zrho_λ, log.(Zrho_λ))
    arr[1] = 0.5 * (relative_entropy + sqrt(4 + relative_entropy^2))
    return arr
end

logdet_pd(W::Hermitian) = logdet(cholesky!(copy(W)))

function new_herm(w, dW::Int, T::Type{<:Real})
    W = similar(w, dW, dW)
    Cones.svec_to_smat!(W, w, sqrt(T(2)))
    return Hermitian(W, :U)
end

function new_herm(w, dW::Int, R::Type{Complex{T}}) where {T<:Real}
    W = zeros(Complex{eltype(w)}, dW, dW)
    Cones.svec_to_smat!(W, w, sqrt(T(2)))
    return Hermitian(W, :U)
end

function von_neumann_entropy(rho)
    λ = eigvals(rho)
    return -sum(λ .* log.(λ))
end

function barrier(point, cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    u = point[1]
    rhoH = new_herm(point[cone.rho_idxs], cone.d, R)
    GrhoH = new_herm(cone.G * point[cone.rho_idxs], cone.d, R)
    ZrhoH = new_herm(cone.Z * point[cone.rho_idxs], cone.d, R)
    relative_entropy = -von_neumann_entropy(GrhoH) + von_neumann_entropy(ZrhoH)
    return -real(log(u - relative_entropy)) - logdet_pd(rhoH)
end

function update_feas(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.feas_updated
    point = cone.point
    @views rho_vec = point[cone.rho_idxs]

    cone.is_feas = false

    svec_to_smat!(cone.rho, rho_vec, cone.rt2)

    if cone.is_G_identity
        cone.Grho = cone.rho
    else
        svec_to_smat!(cone.Grho, cone.G * rho_vec, cone.rt2)
    end
    svec_to_smat!(cone.Zrho, cone.Z * rho_vec, cone.rt2)

    rhoH = Hermitian(cone.rho)

    if isposdef(rhoH)
        cone.Grho_fact = eigen(Hermitian(cone.Grho))
        @. cone.Grho_λ_log = log(cone.Grho_fact.values)
        cone.Zrho_fact = eigen(Hermitian(cone.Zrho))
        @. cone.Zrho_λ_log = log(cone.Zrho_fact.values)
        relative_entropy = dot(cone.Grho_fact.values, cone.Grho_λ_log) - dot(cone.Zrho_fact.values, cone.Zrho_λ_log)
        cone.z = point[1] - relative_entropy
        cone.is_feas = (cone.z > 0)
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert cone.is_feas
    rt2 = cone.rt2
    point = cone.point
    g = cone.grad
    dzdrho = cone.dzdrho

    zi = inv(cone.z)
    g[1] = -zi

    spectral_outer!(cone.Grho_log, cone.Grho_fact.vectors, cone.Grho_λ_log, cone.Gmat)
    smat_to_svec!(cone.Gvec, I(cone.Gd) + cone.Grho_log, rt2)
    cone.is_G_identity ? dzdrho .= cone.Gvec : mul!(dzdrho, cone.Gadj, cone.Gvec)

    spectral_outer!(cone.Zrho_log, cone.Zrho_fact.vectors, cone.Zrho_λ_log, cone.Zmat)
    smat_to_svec!(cone.Zvec, I(cone.Zd) + cone.Zrho_log, rt2)
    mul!(dzdrho, cone.Zadj, cone.Zvec, true, T(-1))

    @views g[cone.rho_idxs] .= -zi * dzdrho

    cone.rho_fact = cone.is_G_identity ? cone.Grho_fact : eigen(Hermitian(cone.rho))
    (rho_λ, rho_vecs) = cone.rho_fact
    spectral_outer!(cone.rho_inv, rho_vecs, inv.(rho_λ), cone.mat) #carefull rho_inv is not exactly Hermitian
    smat_to_svec!(cone.vec, cone.rho_inv, rt2)
    @views g[cone.rho_idxs] .-= cone.vec

    #    fd_grad = ForwardDiff.gradient(p->barrier(p,cone),cone.point)

    #    display("analytic:")
    #    display(new_herm(cleanup!(g[cone.rho_idxs]),cone.d,R))
    #    display("ForwardDiff:")
    #    display(new_herm(cleanup!(fd_grad[cone.rho_idxs]),cone.d,R))

    #backend = AD.FiniteDifferencesBackend(FiniteDifferences.forward_fdm(5, 1))
    #fd_grad_old = AD.gradient(backend, barrier_old, cone.point)[1]

    #    g .= fd_grad
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    rt2 = cone.rt2
    dzdrho = cone.dzdrho
    d2zdrho2 = cone.d2zdrho2
    d2zdrho2G = cone.d2zdrho2G
    d2zdrho2Z = cone.d2zdrho2Z
    rho_idxs = cone.rho_idxs

    (Grho_λ, Grho_vecs) = cone.Grho_fact
    (Zrho_λ, Zrho_vecs) = cone.Zrho_fact

    zi = inv(cone.z)

    H[1, 1] = abs2(zi)
    @. @views H[1, rho_idxs] = abs2(zi) * dzdrho

    @views Hrho = H[rho_idxs, rho_idxs]
    symm_kron!(Hrho, cone.rho_inv, rt2)
    mul!(Hrho, dzdrho, dzdrho', abs2(zi), true)

    Δ2!(cone.Δ2G, Grho_λ, cone.Grho_λ_log)
    eig_dot_kron!(d2zdrho2G, cone.Δ2G, Grho_vecs, cone.Gmat, cone.Gmat2, cone.Gmat3, rt2)
    Δ2!(cone.Δ2Z, Zrho_λ, cone.Zrho_λ_log)
    eig_dot_kron!(d2zdrho2Z, cone.Δ2Z, Zrho_vecs, cone.Zmat, cone.Zmat2, cone.Zmat3, rt2)

    if cone.is_G_identity
        d2zdrho2 .= d2zdrho2G
    else
        spectral_outer!(d2zdrho2, cone.Gadj, Symmetric(d2zdrho2G), cone.big_Gmat)
    end
    mul!(cone.big_Zmat, cone.Zadj, Symmetric(d2zdrho2Z))
    mul!(d2zdrho2, cone.big_Zmat, cone.Z, true, T(-1))
    @. Hrho -= zi * d2zdrho2

    #    d2zdrho2 .= -cone.G'*d2zdrho2G*cone.G + cone.Z'*d2zdrho2Z*cone.Z
    #    @. Hrho -= zi*d2zdrho2

    #    fd_hess = ForwardDiff.hessian(p->barrier(p,cone),cone.point)

    #    display("analytic:")
    #    display(diag(Symmetric(H))[2:end])
    #    display("ForwardDiff:")
    #    display(diag(fd_hess))

    cone.hess_updated = true
    return cone.hess
end

#function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiQKDTri{T, R}) where {T <: Real, R <: RealOrComplex{T}}
#    @assert is_feas(cone)
#
#    @inbounds for i in 1:size(arr, 2)
#        view(prod, :, i) .= ForwardDiff.gradient(
#            s -> ForwardDiff.derivative(t -> barrier(s + t * view(arr, :, i),cone), 0),
#            cone.point,
#        )
#    end
#    
#    return prod
#end

"""
Converts a vector of Kraus operators `K` into a matrix M such that
svec(sum(K[i]*X*K[i]')) == M*svec(X)
for Hermitian matrices X
"""
function kraus2matrix(K::Vector, T::Type, R::Type)
    K = [R.(Ki) for Ki in K]
    return sum(skron.(K))
end

function skron(X)
    R = eltype(X)
    T = real(R)
    iscomplex = R <: Complex
    dout, din = size(X)
    sdout, sdin = Cones.svec_length.(Ref(R), size(X))
    result = Matrix{T}(undef, sdout, sdin)
    symm_kron_full!(result, X, sqrt(T(2)))
    return result
end

function svec(M)
    d = size(M, 1)
    R = eltype(M)
    T = real(R)
    vec_dim = Cones.svec_length(R, d)
    v = Vector{T}(undef, vec_dim)
    return Cones.smat_to_svec!(v, M, sqrt(T(2)))
end

function smat(v, R)
    T = eltype(v)
    R = R <: Complex ? Complex{T} : T
    d = Cones.svec_side(R, length(v))
    M = Matrix{R}(undef, d, d)
    return Cones.svec_to_smat!(M, v, sqrt(T(2)))
end

"""
Computes `skr` such that `skr*svec(x) = svec(mat*x*mat')` for real `mat`
"""
function symm_kron_full!(skr::AbstractMatrix{T}, mat::AbstractMatrix{T}, rt2::T) where {T<:Real}
    dout, din = size(mat)

    col_idx = 1
    @inbounds for l = 1:din
        for k = 1:(l-1)
            row_idx = 1
            for j = 1:dout
                for i = 1:(j-1)
                    skr[row_idx, col_idx] = mat[i, k] * mat[j, l] + mat[i, l] * mat[j, k]
                    row_idx += 1
                end
                skr[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
                row_idx += 1
            end
            col_idx += 1
        end

        row_idx = 1
        for j = 1:dout
            for i = 1:(j-1)
                skr[row_idx, col_idx] = rt2 * mat[i, l] * mat[j, l]
                row_idx += 1
            end
            skr[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
        end
        col_idx += 1
    end

    return skr
end

"""
Computes `skr` such that `skr*svec(x) = svec(mat*x*mat')` for complex `mat`
"""
function symm_kron_full!(skr::AbstractMatrix{T}, mat::AbstractMatrix{Complex{T}}, rt2::T) where {T<:Real}
    dout, din = size(mat)

    col_idx = 1
    @inbounds for l = 1:din
        for k = 1:(l-1)
            row_idx = 1
            for j = 1:dout
                for i = 1:(j-1)
                    a = mat[i, k] * conj(mat[j, l])
                    b = conj(mat[i, l]) * mat[j, k]
                    Cones.spectral_kron_element!(skr, row_idx, col_idx, a, b)
                    row_idx += 2
                end
                c = rt2 * mat[j, k] * conj(mat[j, l])
                skr[row_idx, col_idx] = real(c)
                skr[row_idx, col_idx+1] = imag(c)
                row_idx += 1
            end
            col_idx += 2
        end

        row_idx = 1
        for j = 1:dout
            for i = 1:(j-1)
                c = rt2 * mat[i, l] * conj(mat[j, l])
                skr[row_idx, col_idx] = real(c)
                skr[row_idx+1, col_idx] = -imag(c)
                row_idx += 2
            end
            skr[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
        end
        col_idx += 1
    end

    return skr
end
