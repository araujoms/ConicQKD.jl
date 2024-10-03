mutable struct EpiQKDTri{T<:Real,R<:RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    Gd::Int
    Zd::Int
    is_complex::Bool
    nblocks::Int
    blocks::Vector
    blocksizes::Vector{Int}
    blocknpars::Vector{Int}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}

    Hρ::Matrix{T}
    Hρ_fact::Cholesky{T,Matrix{T}}

    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    hess_aux_updated::Bool
    inv_hess_aux_updated::Bool
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
    ρ_dim::Int
    Gρ_dim::Int
    ρ_idxs::UnitRange{Int}
    ρ::Matrix{R}
    Gρ::Matrix{R}
    Zρ::Vector{Matrix{R}}
    G::Matrix{T}
    Z::Vector{Matrix{T}}
    Gk::Vector{Matrix{R}}
    Zk::Vector{Vector{Matrix{R}}}
    Gadj::Matrix{T}
    Zadj::Vector{Matrix{T}}
    ρ_fact::Eigen{R}
    Gρ_fact::Eigen{R}
    Zρ_fact::Vector{Eigen{R}}
    ρ_inv::Matrix{R}
    ρ_λ_inv::Vector{T}
    Gρ_λ_log::Vector{T}
    Zρ_λ_log::Vector{Vector{T}}
    z::T
    Δ2G::Matrix{T}
    Δ2Z::Vector{Matrix{T}}
    Δ3G::Array{T,3}
    Δ3Z::Vector{Array{T,3}}
    dzdρ::Vector{T}
    d2zdρ2vec::Vector{T}
    d2zdρ2::Matrix{T}
    d2zdρ2G::Matrix{T}
    d2zdρ2Z::Vector{Matrix{T}}
    #variables below are just scratch space
    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    Gmat::Matrix{R}
    Gmat2::Matrix{R}
    Gmat3::Matrix{R}
    Gmat4::Matrix{R}
    Zmat::Vector{Matrix{R}}
    Zmat2::Vector{Matrix{R}}
    Zmat3::Vector{Matrix{R}}
    Zmat4::Vector{Matrix{R}}

    big_Gmat::Matrix{T}
    big_Zmat::Vector{Matrix{T}}
    vec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{Vector{T}}

    function EpiQKDTri{T,R}(
        Gkraus::Vector{<:AbstractMatrix},
        Zkraus::Vector{<:AbstractMatrix},
        dim::Int;
        blocks::Vector{<:AbstractVector} = [1:size(Zkraus[1], 1)],
        use_dual::Bool = false
    ) where {T<:Real,R<:RealOrComplex{T}}
        @assert dim > 1
        cone = new{T,R}()
        cone.use_dual_barrier = use_dual
        cone.blocks = blocks
        cone.nblocks = length(blocks)
        cone.blocksizes = length.(blocks)
        cone.blocknpars = Cones.svec_length.(Ref(R), cone.blocksizes)
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.ρ_dim = dim - 1
        cone.d = size(Gkraus[1], 2)
        cone.Gd = size(Gkraus[1], 1)
        cone.Zd = size(Zkraus[1], 1)
        Gkraus = [R.(Gk) for Gk in Gkraus]
        Zkraus = [R.(Zk) for Zk in Zkraus]
        cone.Gk = Gkraus
        cone.Zk = [filter!(!iszero, [Zk[blocks[i], :] for Zk in Zkraus]) for i = 1:cone.nblocks]
        cone.G = kraus2matrix(Gkraus)
        cone.Z = [kraus2matrix([Zk[blocks[i], :] for Zk in Zkraus]) for i = 1:length(blocks)] #necessary because we can't guarantee that block i is produced only by kraus operator i
        cone.Gadj = Matrix(cone.G')
        cone.Zadj = Matrix.(adjoint.(cone.Z))
        cone.is_G_identity = (cone.Gk == [I(cone.d)])
        cone.Gρ_dim = size(cone.G, 1)
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
                        cone.inv_hess_updated =
                            cone.hess_fact_updated = cone.dder3_aux_updated = cone.inv_hess_aux_updated = false
    )
end

function setup_extra_data!(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    ρ_dim = cone.ρ_dim
    Gρ_dim = cone.Gρ_dim
    d = cone.d
    Gd = cone.Gd
    Zd = cone.Zd
    cone.rt2 = sqrt(T(2))
    cone.ρ_idxs = 2:(ρ_dim+1)
    cone.ρ = zeros(R, d, d)
    cone.Gρ = zeros(R, Gd, Gd)
    cone.Zρ = [zeros(R, s, s) for s in cone.blocksizes]
    cone.ρ_inv = zeros(R, d, d)
    cone.dzdρ = zeros(T, ρ_dim)
    cone.d2zdρ2vec = zeros(T, ρ_dim)
    cone.Δ2G = zeros(T, Gd, Gd)
    cone.Δ2Z = [zeros(T, s, s) for s in cone.blocksizes]
    cone.Δ3G = zeros(T, Gd, Gd, Gd)
    cone.Δ3Z = [zeros(T, s, s, s) for s in cone.blocksizes]
    cone.d2zdρ2 = zeros(T, ρ_dim, ρ_dim)
    cone.d2zdρ2G = zeros(T, Gρ_dim, Gρ_dim)
    cone.d2zdρ2Z = [zeros(T, s, s) for s in cone.blocknpars]
    cone.ρ_λ_inv = zeros(T, d)
    cone.Gρ_λ_log = zeros(T, Gd)
    cone.Zρ_λ_log = [zeros(T, s) for s in cone.blocksizes]

    cone.Hρ = zeros(T, ρ_dim, ρ_dim)

    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.Gmat = zeros(R, Gd, Gd)
    cone.Gmat2 = zeros(R, Gd, Gd)
    cone.Gmat3 = zeros(R, Gd, Gd)
    cone.Gmat4 = zeros(R, Gd, d)
    cone.Zmat = [zeros(R, s, s) for s in cone.blocksizes]
    cone.Zmat2 = [zeros(R, s, s) for s in cone.blocksizes]
    cone.Zmat3 = [zeros(R, s, s) for s in cone.blocksizes]
    cone.Zmat4 = [zeros(R, s, d) for s in cone.blocksizes]
    cone.big_Gmat = zeros(T, ρ_dim, Gρ_dim)
    cone.big_Zmat = [zeros(T, ρ_dim, s) for s in cone.blocknpars]
    cone.vec = zeros(T, ρ_dim)
    cone.Gvec = zeros(T, Gρ_dim)
    cone.Zvec = [zeros(T, s) for s in cone.blocknpars]
    return
end

get_nu(cone::EpiQKDTri) = cone.d + 1

function set_initial_point!(arr::AbstractVector{T}, cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    rt2 = cone.rt2

    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    for i = 1:(cone.d)
        arr[1+k] = 1
        k += incr * i + 1
    end
    @views ρ_vec = arr[cone.ρ_idxs]
    svec_to_smat!(cone.ρ, ρ_vec, rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gmat4)
    end
    applykraus!.(cone.Zρ, cone.Zk, Ref(Hermitian(cone.ρ)), cone.Zmat4)
    Gρ_λ = eigvals(Hermitian(cone.Gρ))
    @. cone.Gρ_λ_log = log(Gρ_λ)
    Zρ_λ = eigvals.(Hermitian.(cone.Zρ))
    for i = 1:cone.nblocks
        cone.Zρ_λ_log[i] .= log.(Zρ_λ[i])
    end
    relative_entropy = dot(Gρ_λ, cone.Gρ_λ_log) - sum(dot.(Zρ_λ, cone.Zρ_λ_log))
    arr[1] = 0.5 * (relative_entropy + sqrt(4 + relative_entropy^2))
    return arr
end

function update_feas(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.feas_updated
    point = cone.point
    @views ρ_vec = point[cone.ρ_idxs]
    rt2 = cone.rt2

    cone.is_feas = false

    svec_to_smat!(cone.ρ, ρ_vec, rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gmat4)
    end
    applykraus!.(cone.Zρ, cone.Zk, Ref(Hermitian(cone.ρ)), cone.Zmat4)

    if isposdef(Hermitian(cone.ρ))
        cone.Gρ_fact = eigen(Hermitian(cone.Gρ))
        cone.Zρ_fact = eigen.(Hermitian.(cone.Zρ))
        if isposdef(cone.Gρ_fact) && all(isposdef.(cone.Zρ_fact)) #necessary because of numerical error
            Gρ_λ = cone.Gρ_fact.values
            Zρ_λ = [fact.values for fact in cone.Zρ_fact]
            @. cone.Gρ_λ_log = log(Gρ_λ)
            for i = 1:cone.nblocks
                cone.Zρ_λ_log[i] .= log.(Zρ_λ[i])
            end
            relative_entropy = dot(Gρ_λ, cone.Gρ_λ_log) - sum(dot.(Zρ_λ, cone.Zρ_λ_log))
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
    Zk = cone.Zk
    Gk = cone.Gk
    dzdρ = cone.dzdρ
    Gvec = cone.Gvec
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat4 = cone.Gmat4
    Zvec = cone.Zvec
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat4 = cone.Zmat4
    blocksizes = cone.blocksizes

    zi = inv(cone.z)
    g[1] = -zi

    Zρ_U = [fact.vectors for fact in cone.Zρ_fact]
    spectral_outer!.(Zmat2, Zρ_U, cone.Zρ_λ_log, Zmat)
    for i = 1:cone.nblocks
        for j = 1:blocksizes[i]
            Zmat2[i][j, j] += 1
        end
    end
    applykraus_adj!(cone.mat2, Zk[1], Hermitian(Zmat2[1]), Zmat4[1])
    for i = 2:cone.nblocks
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat2[i]), Zmat4[1])
        cone.mat2 .+= cone.mat3
    end
    smat_to_svec!(dzdρ, cone.mat2, rt2)

    spectral_outer!(Gmat2, cone.Gρ_fact.vectors, cone.Gρ_λ_log, Gmat)
    for i = 1:cone.Gd
        Gmat2[i, i] += 1
    end
    if cone.is_G_identity
        smat_to_svec!(cone.vec, Gmat2, rt2)
    else
        applykraus_adj!(cone.mat2, Gk, Hermitian(Gmat2), Gmat4)
        smat_to_svec!(cone.vec, cone.mat2, rt2)
    end
    dzdρ .-= cone.vec

    @. @views g[cone.ρ_idxs] = -zi * dzdρ

    cone.ρ_fact = cone.is_G_identity ? cone.Gρ_fact : eigen(Hermitian(cone.ρ))
    (ρ_λ, ρ_U) = cone.ρ_fact
    cone.ρ_λ_inv .= inv.(ρ_λ)
    spectral_outer!(cone.ρ_inv, ρ_U, cone.ρ_λ_inv, cone.mat)
    smat_to_svec!(cone.vec, cone.ρ_inv, rt2)
    @views g[cone.ρ_idxs] .-= cone.vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiQKDTri)
    @assert cone.grad_updated

    Δ2!(cone.Δ2G, cone.Gρ_fact.values, cone.Gρ_λ_log)   #Γ(Λ_G)

    Zρ_λ = [fact.values for fact in cone.Zρ_fact]
    Δ2!.(cone.Δ2Z, Zρ_λ, cone.Zρ_λ_log)   #Γ(Λ_Z)

    return cone.hess_aux_updated = true
end

function d2zdρ2!(
    d2zdρ2vec::AbstractVector{T},
    ρ_arr_mat::AbstractMatrix{R},
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    rt2 = cone.rt2
    Gvec = cone.Gvec
    Zvec = cone.Zvec
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    Zmat4 = cone.Zmat4
    Gρ_U = cone.Gρ_fact.vectors
    Zk = cone.Zk
    Gk = cone.Gk

    # Code corresponding to Z
    applykraus!.(Zmat, Zk, Ref(Hermitian(ρ_arr_mat)), Zmat4)  # Zmat = Z(ξ)
    Zρ_U = [fact.vectors for fact in cone.Zρ_fact]
    spectral_outer!.(Zmat2, adjoint.(Zρ_U), Hermitian.(Zmat), Zmat3) # (U'_Z Z(ξ)U_Z)
    for i = 1:cone.nblocks
        Zmat[i] .= cone.Δ2Z[i] .* Zmat2[i] # Γ(Λ)∘(U'_Z Z(ξ)U_Z)
    end
    spectral_outer!.(Zmat2, Zρ_U, Hermitian.(Zmat), Zmat3)  # U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z
    applykraus_adj!(cone.mat2, Zk[1], Hermitian(Zmat2[1]), Zmat4[1]) # Z'{U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z}
    for i = 2:cone.nblocks
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat2[i]), Zmat4[1])
        cone.mat2 .+= cone.mat3
    end
    smat_to_svec!(d2zdρ2vec, cone.mat2, rt2)

    # Code corresponding to G
    if cone.is_G_identity
        Gmat .= ρ_arr_mat
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), Gmat4)
    end # Gmat = G(ξ)
    spectral_outer!(Gmat2, Gρ_U', Hermitian(Gmat), Gmat3) # (U'_G G(ξ)U_G)
    Gmat .= cone.Δ2G .* Gmat2  # Γ(Λ)∘(U'_G G(ξ)U_G)
    spectral_outer!(Gmat2, Gρ_U, Hermitian(Gmat), Gmat3) # U_G[Γ(Λ_G)∘(U'_G G(ξ)U_G)]U'_G

    if cone.is_G_identity
        smat_to_svec!(cone.vec, Gmat2, rt2)
    else
        applykraus_adj!(Gmat, Gk, Hermitian(Gmat2), Gmat4)
        smat_to_svec!(cone.vec, Gmat, rt2)
    end
    d2zdρ2vec .-= cone.vec # Z'{U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z} - G'{U_G[Γ(Λ)∘(U'_G G(ξ)U_G)]U'_G}

    return d2zdρ2vec
end

"""Multiply the Hessian times the vector ξ. This is more efficient than calculating the Hessian."""
function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    cone.hess_aux_updated || update_hess_aux(cone)

    rt2 = cone.rt2
    ρ_idxs = cone.ρ_idxs
    dzdρ = cone.dzdρ
    d2zdρ2vec = cone.d2zdρ2vec
    ρ_arr_mat = cone.mat
    (ρ_λ, ρ_U) = cone.ρ_fact

    zi = inv(cone.z)

    # For each vector ξ do:
    @inbounds for i = 1:size(arr, 2)
        # Hhh * a_h + Hhρ * a_ρ
        @views ρ_arr = arr[ρ_idxs, i]
        @views ρ_prod = prod[ρ_idxs, i]
        prod[1, i] = abs2(zi) * (arr[1, i] + dot(dzdρ, ρ_arr))  # ξ[1]/u^2 + ⟨∇_ρ(u),ξ[ρ]⟩/u^2

        # Hhρ * a_h + Hρρ * a_ρ
        @. ρ_prod = prod[1, i] * dzdρ

        svec_to_smat!(ρ_arr_mat, ρ_arr, rt2)
        d2zdρ2!(d2zdρ2vec, ρ_arr_mat, cone)

        @. ρ_prod -= zi * d2zdρ2vec

        # Hessian of log(det(ρ))
        spectral_outer!(cone.mat3, ρ_U', Hermitian(ρ_arr_mat), cone.mat2)  # U' ξ U
        ldiv!(Diagonal(ρ_λ), cone.mat3)  # Λ^-1 U' ξ U
        rdiv!(cone.mat3, Diagonal(ρ_λ))  # Λ^-1 U' ξ U Λ^-1
        spectral_outer!(cone.mat3, ρ_U, Hermitian(cone.mat3), cone.mat2)  # U Λ^-1 U' ξ U Λ^-1 U'
        ρ_prod .+= smat_to_svec!(cone.vec, cone.mat3, rt2)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.inv_hess_aux_updated
    @assert cone.grad_updated
    @assert cone.hess_aux_updated

    rt2 = cone.rt2
    dzdρ = cone.dzdρ
    d2zdρ2 = cone.d2zdρ2
    d2zdρ2G = cone.d2zdρ2G
    d2zdρ2Z = cone.d2zdρ2Z
    (Gρ_λ, Gρ_U) = cone.Gρ_fact

    zi = inv(cone.z)

    symm_kron!(cone.Hρ, cone.ρ_inv, rt2) # (ρ⁻¹) ̅ ⊗ρ⁻¹

    eig_dot_kron!(d2zdρ2G, cone.Δ2G, Gρ_U, cone.Gmat, cone.Gmat2, cone.Gmat3, rt2)
    Zρ_U = [fact.vectors for fact in cone.Zρ_fact]
    eig_dot_kron!.(d2zdρ2Z, cone.Δ2Z, Zρ_U, cone.Zmat, cone.Zmat2, cone.Zmat3, Ref(rt2))

    for i = 1:cone.nblocks
        mul!(cone.big_Zmat[i], cone.Zadj[i], d2zdρ2Z[i])
        mul!(d2zdρ2, cone.big_Zmat[i], cone.Z[i], true, i != 1)
    end
    if cone.is_G_identity
        d2zdρ2 .-= d2zdρ2G
    else
        mul!(cone.big_Gmat, cone.Gadj, d2zdρ2G)
        mul!(d2zdρ2, cone.big_Gmat, cone.G, T(-1), true)
    end  # ∇²ᵨᵨu
    @. cone.Hρ -= zi * d2zdρ2 # - 1/u ∇²ᵨᵨu + (ρ⁻¹) ̅ ⊗ρ⁻¹

    cone.Hρ_fact = Hypatia.posdef_fact!(Symmetric(cone.Hρ))
    cone.inv_hess_aux_updated = true
    return
end

#uses the decomposition from appendix B.2 of arXiv:2407.00241
function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    @assert cone.grad_updated

    cone.hess_aux_updated || update_hess_aux(cone)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    ρ_idxs = cone.ρ_idxs
    dzdρ = cone.dzdρ

    u = arr[1, :]
    V = arr[ρ_idxs, :]

    @views ldiv!(prod[ρ_idxs, :], cone.Hρ_fact, V .- dzdρ * u')
    prod[1, :] = abs2(cone.z) * u - prod[ρ_idxs, :]' * dzdρ

    return prod
end

function update_dder3_aux(cone::EpiQKDTri)
    @assert !cone.dder3_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)

    Δ3!(cone.Δ3G, cone.Δ2G, cone.Gρ_fact.values)   # Γ2(Λ_G)

    Zρ_λ = [fact.values for fact in cone.Zρ_fact]
    Δ3!.(cone.Δ3Z, cone.Δ2Z, Zρ_λ)   #Γ2(Λ_Z)

    cone.dder3_aux_updated = true
    return
end

function d3zdρ3!(
    d3zdρ3::AbstractVector{T},
    ρ_dir_mat::AbstractMatrix{R},
    cone::EpiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    rt2 = cone.rt2
    Gk = cone.Gk
    Zk = cone.Zk
    Gvec = cone.Gvec
    Zvec = cone.Zvec
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    Zmat4 = cone.Zmat4
    Gρ_U = cone.Gρ_fact.vectors

    # Code corresponding to Z
    applykraus!.(Zmat, Zk, Ref(Hermitian(ρ_dir_mat)), Zmat4)  # Zmat = Z(ξ)
    Zρ_U = [fact.vectors for fact in cone.Zρ_fact]
    Zvec_sim = spectral_outer!.(Zmat2, adjoint.(Zρ_U), Hermitian.(Zmat), Zmat3) # (U'_Z Z(ξ)U_Z)

    @inbounds @views for n = 1:cone.nblocks # M_Z(ξ) = 2 ∑_k ξ_ik ξ_kj Γ_ijk(Λ_Z)
        for j = 1:cone.blocksizes[n]
            for i = 1:j
                Zmat[n][i, j] = 2 * dot(Zvec_sim[n][:, i], cone.Δ3Z[n][i, j, :] .* Zvec_sim[n][:, j])
            end
        end
    end
    spectral_outer!.(Zmat2, Zρ_U, Hermitian.(Zmat), Zmat3) # U_Z[M_Z(ξ)]U'_Z
    applykraus_adj!(cone.mat2, Zk[1], Hermitian(Zmat2[1]), Zmat4[1]) # Z'{U_Z[M_Z(ξ)]U'_Z}
    for i = 2:cone.nblocks
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat2[i]), Zmat4[1])
        cone.mat2 .+= cone.mat3
    end
    smat_to_svec!(d3zdρ3, cone.mat2, rt2)

    # Code corresponding to G
    if cone.is_G_identity
        Gmat .= ρ_dir_mat
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_dir_mat), Gmat4)
    end # Gmat = G(ξ)
    Gvec_sim = spectral_outer!(Gmat2, Gρ_U', Hermitian(Gmat), Gmat3) # (U'_G G(ξ)U_G)

    @views Gtempvec = cone.Gmat3[:, 1]
    @inbounds @views for j = 1:(cone.Gd) # M_G(ξ) = 2 ∑_k ξ_ik ξ_kj Γ_ijk(Λ_G)
        for i = 1:j
            Gtempvec .= cone.Δ3G[i, j, :] .* Gvec_sim[:, j]
            Gmat[i, j] = 2 * dot(Gvec_sim[:, i], Gtempvec)
        end
    end

    spectral_outer!(Gmat2, Gρ_U, Hermitian(Gmat), Gmat3) # U_G[M_G(ξ)]U'_G

    if cone.is_G_identity
        smat_to_svec!(cone.vec, Gmat2, rt2)
    else
        applykraus_adj!(Gmat, Gk, Hermitian(Gmat2), Gmat4)
        smat_to_svec!(cone.vec, Gmat, rt2) # Z'{U_Z[M_Z(ξ)]U'_Z} - G'{U_G[M_G(ξ)]U'_G}
    end
    d3zdρ3 .-= cone.vec

    return d3zdρ3
end

function dder3(cone::EpiQKDTri{T,R}, dir::AbstractVector{T}) where {T<:Real,R<:RealOrComplex{T}}
    cone.dder3_aux_updated || update_dder3_aux(cone)
    dder3 = cone.dder3
    rt2 = cone.rt2
    zi = inv(cone.z)
    (ρ_λ, ρ_U) = cone.ρ_fact
    ρ_dir_mat = cone.mat
    d2zdρ2vec = cone.d2zdρ2vec

    @views ρ_dir = dir[cone.ρ_idxs]
    svec_to_smat!(ρ_dir_mat, ρ_dir, rt2)
    d2zdρ2!(d2zdρ2vec, ρ_dir_mat, cone) # ∇ρρ(u) * (:, ξ[ρ])

    const0 = zi * (dir[1] + dot(ρ_dir, cone.dzdρ))  # ξ[1] * zi + ∇ρz⋅ξ[ρ]
    const1 = zi * (abs2(const0) - zi * dot(ρ_dir, d2zdρ2vec) * 0.5)  # zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2

    # u
    dder3[1] = const1  # zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 ξ[1] * ∇ρz⋅ξ[ρ]) - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2

    # ρ
    spectral_outer!(cone.mat2, ρ_U', Hermitian(ρ_dir_mat), cone.mat3)  # U' ξ U
    cone.ρ_λ_inv .= sqrt.(ρ_λ)
    @. cone.mat2 /= cone.ρ_λ_inv' #  U' ξ U sqrt(Λ-1)
    ldiv!(Diagonal(ρ_λ), cone.mat2) # Λ-1 U' ξ U sqrt(Λ-1)
    mul!(cone.mat3, cone.mat2, cone.mat2')  # Λ-1 U' ξ U Λ-1 U' ξ U Λ-1
    spectral_outer!(cone.mat3, ρ_U, Hermitian(cone.mat3), cone.mat2)  # mat2 = U Λ-1 U' ξ U Λ-1 U' ξ U Λ-1 U'
    @views dder3_ρ = dder3[cone.ρ_idxs]
    smat_to_svec!(dder3_ρ, cone.mat3, rt2)
    @. dder3_ρ -= const0 * d2zdρ2vec * zi  # U Λ-1 ξ U Λ-1 U' ξ U Λ-1 U' + d3zdρ3 * zi / 2 - zi^2 * (ξ[1] + ∇ρz⋅ξ[ρ]) * d2zdρ2

    d3zdρ3 = d2zdρ2vec #reusing variable to save memory
    d3zdρ3!(d3zdρ3, ρ_dir_mat, cone)

    @. dder3_ρ += zi * d3zdρ3 * 0.5 # U Λ-1 ξ U Λ-1 U' ξ U Λ-1 U' + d3zdρ3 * zi / 2
    @. dder3_ρ += const1 * cone.dzdρ  # += zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) * dzdρ - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2 * dzdρ

    return dder3  # - 0.5 * ∇^3 barrier[ξ,ξ]
end

"""
Converts a vector of Kraus operators `K` into a matrix M such that
svec(sum(K[i]*X*K[i]')) == M*svec(X)
for Hermitian matrices X
"""
function kraus2matrix(K::Vector)
    return sum(skron.(K))
end

function kraus2matrix(K::Matrix)
    return skron(K)
end

function skron(X)
    R = eltype(X)
    T = real(R)
    dout, din = size.(Ref(X), (1, 2))
    sdout, sdin = Cones.svec_length.(Ref(R), (dout, din))
    result = Matrix{T}(undef, sdout, sdin)
    symm_kron_full!(result, X, sqrt(T(2)))
    return result
end

"""
    svec(M::AbstractMatrix, ::Type{R})

Produces the scaled vectorized version of a Hermitian matrix `M` with coefficient type `R`. The transformation preserves inner products, i.e., ⟨M,N⟩ = ⟨svec(M,R),svec(N,R)⟩.
"""
function svec(M::AbstractMatrix, ::Type{R}) where {R} #the weird stuff here is to make it work with JuMP variables
    d = size(M, 1)
    T = real(R)
    vec_dim = Cones.svec_length(R, d)
    v = Vector{real(eltype(1 * M))}(undef, vec_dim)
    if R <: Real
        Cones.smat_to_svec!(v, 1 * M, sqrt(T(2)))
    else
        Cones._smat_to_svec_complex!(v, M, sqrt(T(2)))
    end
    return v
end
export svec
"""
    smat(v::AbstractVector, ::Type{R})

Maps a vector `v` with coefficient type `R` back into a Hermitian matrix M such that svec(M,`R`) = `v`.
"""
function smat(v::AbstractVector, ::Type{R}) where {R} #the weird stuff here is to make it work with JuMP variables
    d = Cones.svec_side(R, length(v))
    T = real(R)
    matrixeltype = R <: Real ? eltype(1 * v) : typeof(complex(v[1], 0))
    M = Matrix{matrixeltype}(undef, d, d)
    if R <: Real
        Cones.svec_to_smat!(M, 1 * v, sqrt(T(2)))
    else
        Cones._svec_to_smat_complex!(M, v, sqrt(T(2)))
    end
    LinearAlgebra.copytri!(M, 'U', true)
    return Hermitian(M)
end
export smat
"""
Computes `skr` such that `skr*svec(x) = svec(mat*x*mat')` for real `mat` and Hermitian `x`
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
Computes `skr` such that `skr*svec(x) = svec(mat*x*mat')` for complex `mat` and Hermitian `x`
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

#temp must have the same dimensions as K[1]
function applykraus!(result, K, X, temp)
    spectral_outer!(result, K[1], X, temp)
    for i = 2:length(K)
        mul!(temp, K[i], X)
        mul!(result, temp, K[i]', true, true)
    end
    return result
end

#temp must have the same dimensions as K[1]
function applykraus_adj!(result, K, X, temp)
    spectral_outer!(result, K[1]', X, temp)
    for i = 2:length(K)
        mul!(temp, X, K[i])
        mul!(result, K[i]', temp, true, true)
    end
    return result
end
