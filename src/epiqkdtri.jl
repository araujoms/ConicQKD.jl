mutable struct EpiQKDTri{T<:Real,R<:RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    Gd::Int
    Zd::Vector{Int}
    is_complex::Bool
    are_blocks_small::Bool
    nblocks::Int
    blocks::Vector{UnitRange{Int}}

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
    ρ_dim::Int
    Gρ_dim::Int
    Zρ_dim::Vector{Int}
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
    ρ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    Gρ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    Zρ_fact::Vector{Eigen{R,T,Matrix{R},Vector{T}}}
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
    #variables below are just scratch space
    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    Gmat::Matrix{R}
    Gmat2::Matrix{R}
    Gmat3::Matrix{R}
    Gρmat::Matrix{R}
    Zmat::Vector{Matrix{R}}
    Zmat2::Vector{Matrix{R}}
    Zmat3::Vector{Matrix{R}}
    Zρmat::Vector{Matrix{R}}

    big_mat::Matrix{T}
    vec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{Vector{T}}

    big_Gmat::Matrix{T}
    big_Zmat::Vector{Matrix{T}}
    d2zdρ2G::Matrix{T}
    d2zdρ2Z::Vector{Matrix{T}}

    function EpiQKDTri{T,R}(
        Gkraus::Vector{<:AbstractMatrix},
        Zkraus::Vector{<:AbstractMatrix},
        dim::Int;
        blocks::Vector{UnitRange{Int}} = [1:size(Zkraus[1], 1)],
        use_dual::Bool = false
    ) where {T<:Real,R<:RealOrComplex{T}}
        @assert dim > 1
        cone = new{T,R}()
        cone.use_dual_barrier = use_dual
        cone.blocks = blocks
        cone.nblocks = length(blocks)

        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.ρ_dim = dim - 1
        cone.d = size(Gkraus[1], 2)
        cone.Gd = size(Gkraus[1], 1)
        cone.Zd = length.(blocks)
        cone.Gρ_dim = Cones.svec_length(R, cone.Gd)
        cone.Zρ_dim = Cones.svec_length.(Ref(R), cone.Zd)

        Gkraus = [R.(Gk) for Gk ∈ Gkraus]
        Zkraus = [R.(Zk) for Zk ∈ Zkraus]
        cone.Gk = Gkraus
        cone.Zk = [filter!(!iszero, [Zk[blocks[i], :] for Zk ∈ Zkraus]) for i ∈ 1:cone.nblocks]
        cone.is_G_identity = (cone.Gk == [I(cone.d)])
        cone.are_blocks_small = maximum(cone.Zd) <= isqrt(cone.d)
        if cone.are_blocks_small
            cone.G = kraus2matrix(Gkraus)
            cone.Z = [kraus2matrix([Zk[blocks[i], :] for Zk ∈ Zkraus]) for i ∈ 1:length(blocks)]
            cone.Gadj = Matrix(cone.G')
            cone.Zadj = Matrix.(adjoint.(cone.Z))
        end
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
    d = cone.d
    Gd = cone.Gd
    Zd = cone.Zd
    ρ_dim = cone.ρ_dim
    Gρ_dim = cone.Gρ_dim
    Zρ_dim = cone.Zρ_dim

    cone.rt2 = sqrt(T(2))
    cone.ρ_idxs = 2:(ρ_dim+1)
    cone.ρ = zeros(R, d, d)
    cone.Gρ = zeros(R, Gd, Gd)
    cone.Zρ = [zeros(R, s, s) for s ∈ Zd]
    cone.ρ_inv = zeros(R, d, d)
    cone.dzdρ = zeros(T, ρ_dim)
    cone.d2zdρ2vec = zeros(T, ρ_dim)
    cone.Δ2G = zeros(T, Gd, Gd)
    cone.Δ2Z = [zeros(T, s, s) for s ∈ Zd]
    cone.Δ3G = zeros(T, Gd, Gd, Gd)
    cone.Δ3Z = [zeros(T, s, s, s) for s ∈ Zd]
    cone.d2zdρ2 = zeros(T, ρ_dim, ρ_dim)
    cone.ρ_λ_inv = zeros(T, d)
    cone.Gρ_λ_log = zeros(T, Gd)
    cone.Zρ_λ_log = [zeros(T, s) for s ∈ Zd]

    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.Gmat = zeros(R, Gd, Gd)
    cone.Gmat2 = zeros(R, Gd, Gd)
    cone.Gmat3 = zeros(R, Gd, Gd)
    cone.Gρmat = zeros(R, Gd, d)
    cone.Zmat = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat2 = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat3 = [zeros(R, s, s) for s ∈ Zd]
    cone.Zρmat = [zeros(R, s, d) for s ∈ Zd]
    cone.big_mat = zeros(T, ρ_dim, ρ_dim)
    cone.vec = zeros(T, ρ_dim)
    cone.Gvec = zeros(T, Gρ_dim)
    cone.Zvec = [zeros(T, s) for s ∈ Zρ_dim]
    if cone.are_blocks_small
        cone.d2zdρ2G = zeros(T, Gρ_dim, Gρ_dim)
        cone.d2zdρ2Z = [zeros(T, s, s) for s ∈ Zρ_dim]
        cone.big_Gmat = zeros(T, ρ_dim, Gρ_dim)
        cone.big_Zmat = [zeros(T, ρ_dim, s) for s ∈ Zρ_dim]
    end
    return
end

get_nu(cone::EpiQKDTri) = cone.d + 1

function set_initial_point!(arr::AbstractVector{T}, cone::EpiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    rt2 = cone.rt2

    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    for i ∈ 1:(cone.d)
        arr[1+k] = 1
        k += incr * i + 1
    end
    @views ρ_vec = arr[cone.ρ_idxs]
    svec_to_smat!(cone.ρ, ρ_vec, rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gρmat)
    end
    applykraus!.(cone.Zρ, cone.Zk, Ref(Hermitian(cone.ρ)), cone.Zρmat)
    Gρ_λ = eigvals(Hermitian(cone.Gρ))
    @. cone.Gρ_λ_log = log(Gρ_λ)
    Zρ_λ = eigvals.(Hermitian.(cone.Zρ))
    for i ∈ 1:cone.nblocks
        cone.Zρ_λ_log[i] .= log.(Zρ_λ[i])
    end
    relative_entropy = dot(Gρ_λ, cone.Gρ_λ_log) - sum(dot.(Zρ_λ, cone.Zρ_λ_log))
    arr[1] = T(0.5) * (relative_entropy + sqrt(4 + relative_entropy^2))
    return arr
end

function update_feas(cone::EpiQKDTri)
    @assert !cone.feas_updated
    point = cone.point
    @views ρ_vec = point[cone.ρ_idxs]
    rt2 = cone.rt2

    cone.is_feas = false

    svec_to_smat!(cone.ρ, ρ_vec, rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gρmat)
    end
    applykraus!.(cone.Zρ, cone.Zk, Ref(Hermitian(cone.ρ)), cone.Zρmat)

    if isposdef(Hermitian(cone.ρ))
        cone.Gρ_fact = eigen(Hermitian(cone.Gρ))
        cone.ρ_fact = cone.is_G_identity ? cone.Gρ_fact : eigen(Hermitian(cone.ρ))
        cone.Zρ_fact = eigen.(Hermitian.(cone.Zρ))
        if isposdef(cone.ρ_fact) && isposdef(cone.Gρ_fact) && all(isposdef.(cone.Zρ_fact)) #necessary because of numerical error
            Gρ_λ = cone.Gρ_fact.values
            Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
            @. cone.Gρ_λ_log = log(Gρ_λ)
            for i ∈ 1:cone.nblocks
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

function update_grad(cone::EpiQKDTri)
    @assert cone.is_feas
    rt2 = cone.rt2
    g = cone.grad
    Zk = cone.Zk
    Gk = cone.Gk
    dzdρ = cone.dzdρ
    Gvec = cone.Gvec
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gρmat = cone.Gρmat
    Zvec = cone.Zvec
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zρmat = cone.Zρmat
    Zd = cone.Zd

    zi = inv(cone.z)
    g[1] = -zi

    spectral_outer!(Gmat2, cone.Gρ_fact.vectors, cone.Gρ_λ_log, Gmat)
    for i ∈ 1:cone.Gd
        Gmat2[i, i] += 1
    end
    if cone.is_G_identity
        cone.mat .= -1 .* Gmat2
    else
        applykraus_adj!(cone.mat, Gk, Hermitian(Gmat2), Gρmat)
        cone.mat .*= -1
    end

    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    spectral_outer!.(Zmat2, Zρ_U, cone.Zρ_λ_log, Zmat)
    for i ∈ 1:cone.nblocks
        for j ∈ 1:Zd[i]
            Zmat2[i][j, j] += 1
        end
    end
    for i ∈ 1:cone.nblocks
        applykraus_adj!(cone.mat2, Zk[i], Hermitian(Zmat2[i]), Zρmat[1])
        cone.mat .+= cone.mat2
    end
    smat_to_svec!(dzdρ, cone.mat, rt2)

    @. @views g[cone.ρ_idxs] = -zi * dzdρ

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

    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
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
    Gρmat = cone.Gρmat
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    Zρmat = cone.Zρmat
    Gρ_U = cone.Gρ_fact.vectors
    Zk = cone.Zk
    Gk = cone.Gk

    # Code corresponding to G
    if cone.is_G_identity
        Gmat .= ρ_arr_mat
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), Gρmat)
    end # Gmat = G(ξ)
    spectral_outer!(Gmat2, Gρ_U', Hermitian(Gmat), Gmat3) # (U'_G G(ξ)U_G)
    Gmat .= cone.Δ2G .* Gmat2  # Γ(Λ)∘(U'_G G(ξ)U_G)
    spectral_outer!(Gmat2, Gρ_U, Hermitian(Gmat), Gmat3) # U_G[Γ(Λ_G)∘(U'_G G(ξ)U_G)]U'_G

    if cone.is_G_identity
        cone.mat2 .= -1 .* Gmat2
    else
        applykraus_adj!(cone.mat2, Gk, Hermitian(Gmat2), Gρmat)
        cone.mat2 .*= -1
    end

    # Code corresponding to Z
    applykraus!.(Zmat, Zk, Ref(Hermitian(ρ_arr_mat)), Zρmat)  # Zmat = Z(ξ)
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    spectral_outer!.(Zmat2, adjoint.(Zρ_U), Hermitian.(Zmat), Zmat3) # (U'_Z Z(ξ)U_Z)
    for i ∈ 1:cone.nblocks
        Zmat[i] .= cone.Δ2Z[i] .* Zmat2[i] # Γ(Λ)∘(U'_Z Z(ξ)U_Z)
    end
    spectral_outer!.(Zmat2, Zρ_U, Hermitian.(Zmat), Zmat3)  # U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z
    for i ∈ 1:cone.nblocks # Z'{U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z}
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat2[i]), Zρmat[1])
        cone.mat2 .+= cone.mat3
    end
    smat_to_svec!(d2zdρ2vec, cone.mat2, rt2)  # Z'{U_Z[Γ(Λ_Z)∘(U'_Z Z(ξ)U_Z)]U'_Z} - G'{U_G[Γ(Λ)∘(U'_G G(ξ)U_G)]U'_G}

    return d2zdρ2vec
end

"""Multiply the Hessian times the vector ξ. This is more efficient than calculating the Hessian."""
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiQKDTri)
    cone.hess_aux_updated || update_hess_aux(cone)

    ρ_idxs = cone.ρ_idxs
    dzdρ = cone.dzdρ
    ρ_arr_mat = cone.mat
    (ρ_λ, ρ_U) = cone.ρ_fact

    zi = inv(cone.z)

    # For each vector ξ do:
    @inbounds for i ∈ 1:size(arr, 2)
        # Hhh * a_h + Hhρ * a_ρ
        @views ρ_arr = arr[ρ_idxs, i]
        @views ρ_prod = prod[ρ_idxs, i]
        prod[1, i] = abs2(zi) * (arr[1, i] + dot(dzdρ, ρ_arr))  # ξ[1]/u^2 + ⟨∇_ρ(u),ξ[ρ]⟩/u^2

        # Hhρ * a_h + Hρρ * a_ρ
        @. ρ_prod = prod[1, i] * dzdρ

        svec_to_smat!(ρ_arr_mat, ρ_arr, cone.rt2)
        d2zdρ2!(cone.d2zdρ2vec, ρ_arr_mat, cone)

        @. ρ_prod -= zi * cone.d2zdρ2vec

        # Hessian of log(det(ρ))
        spectral_outer!(cone.mat3, ρ_U', Hermitian(ρ_arr_mat), cone.mat2)  # U' ξ U
        ldiv!(Diagonal(ρ_λ), cone.mat3)  # Λ^-1 U' ξ U
        rdiv!(cone.mat3, Diagonal(ρ_λ))  # Λ^-1 U' ξ U Λ^-1
        spectral_outer!(cone.mat3, ρ_U, Hermitian(cone.mat3), cone.mat2)  # U Λ^-1 U' ξ U Λ^-1 U'
        ρ_prod .+= smat_to_svec!(cone.vec, cone.mat3, cone.rt2)
    end

    return prod
end

function update_hess(cone::EpiQKDTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)

    H = cone.hess.data
    Gk = cone.Gk
    Zk = cone.Zk
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    Δ2G = cone.Δ2G
    Δ2Z = cone.Δ2Z
    rt2 = cone.rt2
    dzdρ = cone.dzdρ
    d2zdρ2 = cone.d2zdρ2
    ρ_idxs = cone.ρ_idxs
    Gρ_U = cone.Gρ_fact.vectors
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]

    zi = inv(cone.z)
    H[1, 1] = abs2(zi)
    @. @views H[1, ρ_idxs] = abs2(zi) * dzdρ

    @views Hρ = H[ρ_idxs, ρ_idxs]
    symm_kron!(Hρ, cone.ρ_inv, rt2)
    mul!(Hρ, dzdρ, dzdρ', abs2(zi), true)

    if cone.are_blocks_small #for small blocks it's more efficient to compute the little pieces with hessian_spectral_function! and later expand them into d2zdρ2
        d2zdρ2G = cone.d2zdρ2G
        d2zdρ2Z = cone.d2zdρ2Z

        copyto!(Gmat, Gρ_U')
        d_spectral!(d2zdρ2G, Δ2G, Gmat, Gmat2, Gmat3, rt2)
        if cone.is_G_identity
            d2zdρ2 .= -1 .* d2zdρ2G
        else
            mul!(cone.big_Gmat, cone.Gadj, d2zdρ2G)
            mul!(d2zdρ2, cone.big_Gmat, cone.G, -1, false)
        end

        for i ∈ eachindex(cone.blocks)
            copyto!(Zmat[i], Zρ_U[i]')
        end
        d_spectral!.(d2zdρ2Z, Δ2Z, Zmat, Zmat2, Zmat3, Ref(rt2))
        for i ∈ 1:cone.nblocks
            mul!(cone.big_Zmat[i], cone.Zadj[i], d2zdρ2Z[i])
            mul!(d2zdρ2, cone.big_Zmat[i], cone.Z[i], true, true)
        end

    else
        if length(Gk) == 1
            Gρ_U_adj_Gk = Gρ_U' * Gk[1]
        else
            Gρ_U_adj_Gk = [Gρ_U' * Gki for Gki ∈ Gk]
        end
        d_spectral!(d2zdρ2, Δ2G, Gρ_U_adj_Gk, Gmat2, Gmat3, cone.Gρmat, cone.mat, rt2)
        d2zdρ2 .*= -1

        if all(length.(Zk) .== 1)
            Zρ_U_adj_Zk = [Zρ_U[i]' * Zk[i][1] for i ∈ 1:cone.nblocks]
        else
            Zρ_U_adj_Zk = [[Zρ_U[i]' * Zk[i][j] for j ∈ 1:length(Zk[i])] for i ∈ 1:cone.nblocks]
        end
        for i ∈ 1:cone.nblocks
            d_spectral!(cone.big_mat, Δ2Z[i], Zρ_U_adj_Zk[i], Zmat2[i], Zmat3[i], cone.Zρmat[i], cone.mat, rt2)
            d2zdρ2 .+= cone.big_mat
        end
    end

    @. Hρ -= zi * d2zdρ2

    cone.hess_updated = true
    return cone.hess
end

function update_dder3_aux(cone::EpiQKDTri)
    @assert !cone.dder3_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)

    Δ3!(cone.Δ3G, cone.Δ2G, cone.Gρ_fact.values)   # Γ2(Λ_G)

    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
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
    Gρmat = cone.Gρmat
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    Zρmat = cone.Zρmat
    Gρ_U = cone.Gρ_fact.vectors

    # Code corresponding to G
    if cone.is_G_identity
        Gmat .= ρ_dir_mat
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_dir_mat), Gρmat)
    end # Gmat = G(ξ)
    Gvec_sim = spectral_outer!(Gmat2, Gρ_U', Hermitian(Gmat), Gmat3) # (U'_G G(ξ)U_G)

    @views Gtempvec = cone.Gmat3[:, 1]
    @inbounds @views for j ∈ 1:(cone.Gd) # M_G(ξ) = 2 ∑_k ξ_ik ξ_kj Γ_ijk(Λ_G)
        for i ∈ 1:j
            Gtempvec .= cone.Δ3G[i, j, :] .* Gvec_sim[:, j]
            Gmat[i, j] = 2 * dot(Gvec_sim[:, i], Gtempvec)
        end
    end

    spectral_outer!(Gmat2, Gρ_U, Hermitian(Gmat), Gmat3) # U_G[M_G(ξ)]U'_G

    if cone.is_G_identity
        cone.mat2 .= -1 .* Gmat2
    else
        applykraus_adj!(cone.mat2, Gk, Hermitian(Gmat2), Gρmat)
        cone.mat2 .*= -1
    end

    # Code corresponding to Z
    applykraus!.(Zmat, Zk, Ref(Hermitian(ρ_dir_mat)), Zρmat)  # Zmat = Z(ξ)
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    Zvec_sim = spectral_outer!.(Zmat2, adjoint.(Zρ_U), Hermitian.(Zmat), Zmat3) # (U'_Z Z(ξ)U_Z)

    @inbounds @views for n ∈ 1:cone.nblocks # M_Z(ξ) = 2 ∑_k ξ_ik ξ_kj Γ_ijk(Λ_Z)
        for j ∈ 1:cone.Zd[n]
            for i ∈ 1:j
                Zmat[n][i, j] = 2 * dot(Zvec_sim[n][:, i], cone.Δ3Z[n][i, j, :] .* Zvec_sim[n][:, j])
            end
        end
    end
    spectral_outer!.(Zmat2, Zρ_U, Hermitian.(Zmat), Zmat3) # U_Z[M_Z(ξ)]U'_Z
    for i ∈ 1:cone.nblocks # Z'{U_Z[M_Z(ξ)]U'_Z}
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat2[i]), Zρmat[1])
        cone.mat2 .+= cone.mat3
    end
    smat_to_svec!(d3zdρ3, cone.mat2, rt2) # Z'{U_Z[M_Z(ξ)]U'_Z} - G'{U_G[M_G(ξ)]U'_G}

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
    tempvec = cone.ρ_λ_inv
    tempvec .= sqrt.(ρ_λ)
    @. cone.mat2 /= tempvec' #  U' ξ U sqrt(Λ-1)
    ldiv!(Diagonal(ρ_λ), cone.mat2) # Λ-1 U' ξ U sqrt(Λ-1)
    mul!(cone.mat3, ρ_U, cone.mat2) # ρ-1 ξ U sqrt(Λ-1)
    mul!(cone.mat2, cone.mat3, cone.mat3')  # ρ-1 ξ ρ-1 ξ ρ-1
    @views dder3_ρ = dder3[cone.ρ_idxs]
    smat_to_svec!(dder3_ρ, cone.mat2, rt2)
    @. dder3_ρ -= const0 * d2zdρ2vec * zi  # U Λ-1 ξ U Λ-1 U' ξ U Λ-1 U' + d3zdρ3 * zi / 2 - zi^2 * (ξ[1] + ∇ρz⋅ξ[ρ]) * d2zdρ2

    d3zdρ3 = d2zdρ2vec #reusing variable to save memory
    d3zdρ3!(d3zdρ3, ρ_dir_mat, cone)

    @. dder3_ρ += zi * d3zdρ3 * T(0.5) # U Λ-1 ξ U Λ-1 U' ξ U Λ-1 U' + d3zdρ3 * zi / 2
    @. dder3_ρ += const1 * cone.dzdρ  # += zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) * dzdρ - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2 * dzdρ

    return dder3  # - 0.5 * ∇^3 barrier[ξ,ξ]
end
