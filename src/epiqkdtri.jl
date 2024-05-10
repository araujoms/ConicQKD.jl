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
    G::Matrix{T}
    Z::Matrix{T}
    Gadj::Matrix{T}
    Zadj::Matrix{T}
    rho_fact::Eigen{R}
    Grho_fact::Eigen{R}
    Zrho_fact::Eigen{R}
    rho_inv::Matrix{R}
    rho_λ_inv::Vector{T}
    Grho_λ_log::Vector{T}
    Zrho_λ_log::Vector{T}
    z::T
    dzdrho::Vector{T}
    Δ2G::Matrix{T}
    Δ2Z::Matrix{T}
    Δ3G::Array{T,3}
    Δ3Z::Array{T,3}
    d2zdrho2::Matrix{T}
    d2zdrho2G::Matrix{T}
    d2zdrho2Z::Matrix{T}
    #variables below are just scratch space
    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    Gmat::Matrix{R}
    Gmat2::Matrix{R}
    Gmat3::Matrix{R}
    Zmat::Matrix{R}
    Zmat2::Matrix{R}
    Zmat3::Matrix{R}
    big_Gmat::Matrix{T}
    big_Zmat::Matrix{T}
    vec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{T}

    function EpiQKDTri{T,R}(
        Gkraus::Vector,
        Zkraus::Vector,
        dim::Int;
        use_dual::Bool = false
    ) where {T<:Real,R<:RealOrComplex{T}}
        @assert dim > 1
        cone = new{T,R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.rho_dim = dim - 1
        cone.d = svec_side(R, cone.rho_dim)
        cone.Gd = size(Gkraus[1], 1)
        cone.Zd = size(Zkraus[1], 1)
        cone.G = kraus2matrix(Gkraus, R)
        cone.Z = kraus2matrix(Zkraus, R)
        cone.Gadj = Matrix(cone.G')
        cone.Zadj = Matrix(cone.Z')
        cone.is_G_identity = cone.G == I(cone.rho_dim)
        cone.Grho_dim = size(cone.G, 1)
        cone.Zrho_dim = size(cone.Z, 1)
        return cone
    end
end

use_dder3(cone::EpiQKDTri) = true

function reset_data(cone::EpiQKDTri)
    return (
        cone.feas_updated =
            cone.grad_updated =
                cone.hess_updated =
                    cone.hess_aux_updated =
                        cone.inv_hess_updated = cone.hess_fact_updated = cone.dder3_aux_updated = false
    )
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
    cone.rho_inv = zeros(R, d, d)
    cone.dzdrho = zeros(T, rho_dim)
    cone.Δ2G = zeros(T, Gd, Gd)
    cone.Δ2Z = zeros(T, Zd, Zd)
    cone.Δ3G = zeros(T, Gd, Gd, Gd)
    cone.Δ3Z = zeros(T, Zd, Zd, Zd)
    cone.d2zdrho2 = zeros(T, rho_dim, rho_dim)
    cone.d2zdrho2G = zeros(T, Grho_dim, Grho_dim)
    cone.d2zdrho2Z = zeros(T, Zrho_dim, Zrho_dim)
    cone.rho_λ_inv = zeros(T, d)
    cone.Grho_λ_log = zeros(T, Gd)
    cone.Zrho_λ_log = zeros(T, Zd)

    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.Gmat = zeros(R, Gd, Gd)
    cone.Gmat2 = zeros(R, Gd, Gd)
    cone.Gmat3 = zeros(R, Gd, Gd)
    cone.Zmat = zeros(R, Zd, Zd)
    cone.Zmat2 = zeros(R, Zd, Zd)
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
        cone.Zrho_fact = eigen(Hermitian(cone.Zrho))
        if isposdef(cone.Grho_fact) && isposdef(cone.Zrho_fact) #necessary because of numerical error
            Grho_λ = cone.Grho_fact.values
            Zrho_λ = cone.Zrho_fact.values
            @. cone.Grho_λ_log = log(Grho_λ)
            @. cone.Zrho_λ_log = log(Zrho_λ)
            relative_entropy = dot(Grho_λ, cone.Grho_λ_log) - dot(Zrho_λ, cone.Zrho_λ_log)
            cone.z = point[1] - relative_entropy
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert cone.is_feas
    rt2 = cone.rt2
    g = cone.grad
    dzdrho = cone.dzdrho
    Gmat2 = cone.Gmat2
    Zmat2 = cone.Zmat2

    zi = inv(cone.z)
    g[1] = -zi

    spectral_outer!(Gmat2, cone.Grho_fact.vectors, cone.Grho_λ_log, cone.Gmat)
    for i = 1:cone.Gd
        Gmat2[i, i] += 1
    end
    smat_to_svec!(cone.Gvec, Gmat2, rt2)
    cone.is_G_identity ? dzdrho .= cone.Gvec : mul!(dzdrho, cone.Gadj, cone.Gvec)

    spectral_outer!(Zmat2, cone.Zrho_fact.vectors, cone.Zrho_λ_log, cone.Zmat)
    for i = 1:cone.Zd
        Zmat2[i, i] += 1
    end
    smat_to_svec!(cone.Zvec, Zmat2, rt2)
    mul!(dzdrho, cone.Zadj, cone.Zvec, true, T(-1))

    @. @views g[cone.rho_idxs] = -zi * dzdrho

    cone.rho_fact = cone.is_G_identity ? cone.Grho_fact : eigen(Hermitian(cone.rho))
    (rho_λ, rho_vecs) = cone.rho_fact
    cone.rho_λ_inv .= inv.(rho_λ)
    spectral_outer!(cone.rho_inv, rho_vecs, cone.rho_λ_inv, cone.mat) #carefull rho_inv is not exactly Hermitian
    smat_to_svec!(cone.vec, cone.rho_inv, rt2)
    @views g[cone.rho_idxs] .-= cone.vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiQKDTri)
    @assert cone.grad_updated

    Δ2!(cone.Δ2G, cone.Grho_fact.values, cone.Grho_λ_log)   #Γ(Λ_G)
    Δ2!(cone.Δ2Z, cone.Zrho_fact.values, cone.Zrho_λ_log)   #Γ(Λ_Z)

    cone.hess_aux_updated = true
    return cone.hess_aux_updated
end

function update_hess(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    cone.hess_aux_updated || update_hess_aux(cone)
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

    eig_dot_kron!(d2zdrho2G, cone.Δ2G, Grho_vecs, cone.Gmat, cone.Gmat2, cone.Gmat3, rt2)
    eig_dot_kron!(d2zdrho2Z, cone.Δ2Z, Zrho_vecs, cone.Zmat, cone.Zmat2, cone.Zmat3, rt2)

    if cone.is_G_identity
        d2zdrho2 .= d2zdrho2G
    else
        spectral_outer!(d2zdrho2, cone.Gadj, Symmetric(d2zdrho2G), cone.big_Gmat)
    end
    mul!(cone.big_Zmat, cone.Zadj, Symmetric(d2zdrho2Z))
    mul!(d2zdrho2, cone.big_Zmat, cone.Z, true, T(-1))
    @. Hrho -= zi * d2zdrho2

    cone.hess_updated = true
    return cone.hess
end

function d2zdrho2!(
    d2zdρ2::AbstractVector{T},
    rho_dir::AbstractVector{T},
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    rt2 = cone.rt2
    Gvec = cone.Gvec
    Zvec = cone.Zvec
    Gmat = cone.Gmat
    Zmat = cone.Zmat
    Gmat2 = cone.Gmat2
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    # Factorizations of G(ρ), Z(ρ) and ρ
    Grho_vecs = cone.Grho_fact.vectors
    Zrho_vecs = cone.Zrho_fact.vectors

    # Code corresponding to G
    if cone.is_G_identity
        svec_to_smat!(Gmat, rho_dir, rt2)
    else
        mul!(Gvec, cone.G, rho_dir)
        svec_to_smat!(Gmat, Gvec, rt2)
    end # Gmat = G(ξ)
    spectral_outer!(Gmat2, Grho_vecs', Hermitian(Gmat), cone.Gmat3) # (U'_G G(ξ)U_G)
    Gmat .= cone.Δ2G .* Gmat2  # Γ(Λ)∘(U'_G G(ξ)U_G)
    spectral_outer!(Gmat2, Grho_vecs, Hermitian(Gmat), cone.Gmat3) # U_G[Γ(Λ_G)∘(U'_G G(ξ)U_G)]U'_G

    smat_to_svec!(Gvec, Gmat2, rt2)
    if cone.is_G_identity
        d2zdρ2 .= Gvec
    else
        mul!(d2zdρ2, cone.Gadj, Gvec) # G'{U_G[Γ(Λ)∘(U'_G G(ξ)U_G)]U'_G}
    end

    # Code corresponding to Z
    mul!(Zvec, cone.Z, rho_dir)
    svec_to_smat!(Zmat, Zvec, rt2)  # Zmat = Z(ξ)
    spectral_outer!(Zmat2, Zrho_vecs', Hermitian(Zmat), Zmat3) # (U'_Z Z(ξ)U_Z)

    Zmat .= cone.Δ2Z .* Zmat2  # Γ(Λ)∘(U'_Z Z(ξ)U_Z)
    spectral_outer!(Zmat2, Zrho_vecs, Hermitian(Zmat), Zmat3)  # U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z
    smat_to_svec!(Zvec, Zmat2, rt2)
    mul!(d2zdρ2, cone.Zadj, Zvec, 1, -1)  # Z'{U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z} - G'{U_G[Γ(Λ)∘(U'_G G(ξ)U_G)]U'_G}

    return d2zdρ2
end

"""Multiply the Hessian times the vector ξ. This is more efficient than calculating the Hessian."""
function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    cone.hess_aux_updated || update_hess_aux(cone)
    rt2 = cone.rt2
    rho_idxs = cone.rho_idxs
    dzdrho = cone.dzdrho
    d2zdρ2 = cone.vec

    (rho_λ, rho_vecs) = cone.rho_fact

    zi = inv(cone.z)

    # For each vector ξ do:
    @inbounds for i = 1:size(arr, 2)
        # Hhh * a_h + Hhρ * a_ρ
        @views rho_arr = arr[rho_idxs, i]
        @views rho_prod = prod[rho_idxs, i]
        prod[1, i] = abs2(zi) * (arr[1, i] + dot(dzdrho, rho_arr))  # ξ[1]/u^2 + ⟨∇_ρ(u),ξ[ρ]⟩/u^2

        # Hhρ * a_h + Hρρ * a_ρ
        @. rho_prod = prod[1, i] * dzdrho

        d2zdrho2!(d2zdρ2, rho_arr, cone)

        @. rho_prod -= zi * d2zdρ2

        # Hessian of log(det(ρ))
        svec_to_smat!(cone.mat, rho_arr, rt2)  # svec(ξ) -> smat(ξ)
        spectral_outer!(cone.mat, rho_vecs', Hermitian(cone.mat), cone.mat2)  # U' ξ U
        ldiv!(Diagonal(rho_λ), cone.mat)  # Λ^-1 U' ξ U
        rdiv!(cone.mat, Diagonal(rho_λ))  # Λ^-1 U' ξ U Λ^-1
        spectral_outer!(cone.mat, rho_vecs, Hermitian(cone.mat), cone.mat2)  # U Λ^-1 U' ξ U Λ^-1 U'
        rho_prod .+= smat_to_svec!(cone.vec, cone.mat, rt2)
    end

    return prod
end

function update_dder3_aux(cone::EpiQKDTri)
    @assert !cone.dder3_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)

    Δ3!(cone.Δ3G, cone.Δ2G, cone.Grho_fact.values)   # Γ2(Λ_G)
    Δ3!(cone.Δ3Z, cone.Δ2Z, cone.Zrho_fact.values)   # Γ2(Λ_Z)

    cone.dder3_aux_updated = true
    return
end

function d3zdrho3!(
    d3zdρ3::AbstractVector{T},
    rho_dir::AbstractVector{T},
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    rt2 = cone.rt2
    Gvec = cone.Gvec
    Zvec = cone.Zvec
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    # Factorizations of G(ρ), Z(ρ) and ρ
    Grho_vecs = cone.Grho_fact.vectors
    Zrho_vecs = cone.Zrho_fact.vectors

    # Code corresponding to G
    if cone.is_G_identity
        svec_to_smat!(Gmat, rho_dir, rt2)
    else
        mul!(Gvec, cone.G, rho_dir)
        svec_to_smat!(Gmat, Gvec, rt2)
    end # Gmat = G(ξ)
    Gvec_sim = spectral_outer!(Gmat2, Grho_vecs', Hermitian(Gmat), Gmat3) # (U'_G G(ξ)U_G)

    @views Gtempvec = cone.Gmat3[:, 1]
    @inbounds @views for j = 1:(cone.Gd)
        for i = 1:j
            Gtempvec .= cone.Δ3G[i, j, :] .* Gvec_sim[:, j]
            Gmat[i, j] = 2 * dot(Gvec_sim[:, i], Gtempvec)
        end
    end
    # M_G(ξ) = 2 ∑_k ξ_ik ξ_jk Γ_ijk(Λ_G)

    spectral_outer!(Gmat2, Grho_vecs, Hermitian(Gmat), Gmat3) # U_G[M_G(ξ)]U'_G

    smat_to_svec!(Gvec, Gmat2, rt2)
    if cone.is_G_identity
        d3zdρ3 .= Gvec
    else
        mul!(d3zdρ3, cone.Gadj, Gvec) # G'{U_G[M_G(ξ)]U'_G}
    end

    # Code corresponding to Z
    mul!(Zvec, cone.Z, rho_dir)
    svec_to_smat!(Zmat, Zvec, rt2)  # Zmat = Z(ξ)
    Zvec_sim = spectral_outer!(Zmat2, Zrho_vecs', Hermitian(Zmat), Zmat3) # (U'_Z Z(ξ)U_Z)

    @views Ztempvec = cone.Zmat3[:, 1]
    @inbounds @views for j = 1:(cone.Zd)
        for i = 1:j
            Ztempvec .= cone.Δ3Z[i, j, :] .* Zvec_sim[:, j]
            Zmat[i, j] = 2 * dot(Zvec_sim[:, i], Ztempvec)
        end
    end
    # M_Z(ξ) = 2 ∑_k ξ_ik ξ_jk Γ_ijk(Λ_Z)

    spectral_outer!(Zmat2, Zrho_vecs, Hermitian(Zmat), Zmat3)  # U_Z[M_Z(ξ)]U'_Z
    smat_to_svec!(Zvec, Zmat2, rt2)
    mul!(d3zdρ3, cone.Zadj, Zvec, 1, -1)  # Z'{U_Z[M_Z(ξ)]U'_Z} - G'{U_G[M_G(ξ)]U'_G}

    return d3zdρ3
end

function dder3(cone::EpiQKDTri{T,R}, dir::AbstractVector{T}) where {T<:Real,R<:RealOrComplex{T}}
    cone.dder3_aux_updated || update_dder3_aux(cone)
    dder3 = cone.dder3
    rt2 = cone.rt2
    zi = inv(cone.z)
    (rho_λ, U) = cone.rho_fact

    @views rho_dir = dir[cone.rho_idxs]
    ddu = d2zdrho2!(cone.vec, rho_dir, cone) # ∇ρρ(u) * (:, ξ[ρ])

    const0 = zi * (dir[1] + dot(rho_dir, cone.dzdrho))  # ξ[1] * zi + ∇ρz⋅ξ[ρ]
    const1 = zi * (abs2(const0) - zi * dot(rho_dir, ddu) * 0.5)  # zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2

    # u
    dder3[1] = const1  # zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 ξ[1] * ∇ρz⋅ξ[ρ]) - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2

    # ρ
    svec_to_smat!(cone.mat, rho_dir, rt2) # ξ -> svec(ξ)
    spectral_outer!(cone.mat2, U', Hermitian(cone.mat), cone.mat3)  # U' ξ U
    cone.rho_λ_inv .= sqrt.(rho_λ)
    @. cone.mat2 /= cone.rho_λ_inv' #  U' ξ U sqrt(Λ-1)
    ldiv!(Diagonal(rho_λ), cone.mat2) # Λ-1 U' ξ U sqrt(Λ-1)
    mul!(cone.mat, cone.mat2, cone.mat2')  # Λ-1 U' ξ U Λ-1 U' ξ U Λ-1
    spectral_outer!(cone.mat2, U, Hermitian(cone.mat), cone.mat3)  # mat2 = U Λ-1 U' ξ U Λ-1 U' ξ U Λ-1 U'
    @views dder3_rho = dder3[cone.rho_idxs]
    smat_to_svec!(dder3_rho, cone.mat2, rt2)
    @. dder3_rho -= const0 * ddu * zi  # U Λ-1 ξ U Λ-1 U' ξ U Λ-1 U' + d3zdρ3 * zi / 2 - zi^2 * (ξ[1] + ∇ρz⋅ξ[ρ]) * d2zdρ2

    d3zdrho3!(cone.vec, rho_dir, cone)
    @. dder3_rho += zi * cone.vec * 0.5 # U Λ-1 ξ U Λ-1 U' ξ U Λ-1 U' + d3zdρ3 * zi / 2
    @. dder3_rho += const1 * cone.dzdrho  # += zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) * dzdρ - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2 * dzdρ

    return dder3  # - 0.5 * ∇^3 barrier[ξ,ξ]
end

"""
Converts a vector of Kraus operators `K` into a matrix M such that
svec(sum(K[i]*X*K[i]')) == M*svec(X)
for Hermitian matrices X
"""
function kraus2matrix(K::Vector, R::Type)
    K = [R.(Ki) for Ki in K]
    return sum(skron.(K))
end

function skron(X)
    R = eltype(X)
    T = real(R)
    iscomplex = R <: Complex
    dout, din = size.(Ref(X), (1, 2))
    sdout, sdin = Cones.svec_length.(Ref(R), (dout, din))
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
function symm_kron_full!(skr::AbstractMatrix{T}, mat::AbstractVecOrMat{T}, rt2::T) where {T<:Real}
    dout, din = size.(Ref(mat), (1, 2))

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
function symm_kron_full!(skr::AbstractMatrix{T}, mat::AbstractVecOrMat{Complex{T}}, rt2::T) where {T<:Real}
    dout, din = size.(Ref(mat), (1, 2))

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
