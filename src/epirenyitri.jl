mutable struct EpiRenyiTri{T<:Real,R<:RealOrComplex{T}} <: Cone{T}
    α::T
    α2::T
    sα::Int
    use_dual_barrier::Bool
    dim::Int
    d::Int
    Gd::Int
    Zd::Vector{Int}
    ZD::Int
    is_complex::Bool
    are_blocks_small::Bool
    nblocks::Int
    blocks::Vector

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
    is_S_identity::Bool
    ρ_dim::Int
    Gρ_dim::Int
    Zρ_dim::Vector{Int}
    ρ_idxs::UnitRange{Int}
    ρ::Matrix{R}
    Gρ::Matrix{R}
    Gρroot::Matrix{R}
    Zρ::Vector{Matrix{R}}
    Zρα::Vector{Matrix{R}}
    G::Matrix{T}
    S::Union{Matrix{R},UniformScaling{Bool}}
    Z::Vector{Matrix{T}}
    Gk::Vector{Matrix{R}}
    Zk::Vector{Vector{Matrix{R}}}
    Zkbig::Vector{Matrix{R}}
    Gadj::Matrix{T}
    Zadj::Vector{Matrix{T}}
    ρ_fact::Eigen{R}
    Gρ_fact::Eigen{R}
    Zρ_fact::Vector{Eigen{R}}
    ZG_fact::SVD{R,T,Matrix{R},Vector{T}}
    ρ_inv::Matrix{R}
    ρ_λ_inv::Vector{T}
    Gρ_λ_log::Vector{T}
    Zρ_λ_log::Vector{Vector{T}}
    Zρα_λ::Vector{Vector{T}}
    z::T
    Δ2G::Matrix{T}
    Δ2Z::Vector{Matrix{T}}
    Δ3G::Array{T,3}
    Δ3Z::Vector{Array{T,3}}
    dzdρ::Vector{T}
    d2zdρ2vec::Vector{T}
    d2zdρ2::Matrix{T}

    ZG::Matrix{R}
    ZS::Matrix{R}
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
    ZGmat::Matrix{R}
    ZGmat2::Matrix{R}

    big_mat::Matrix{T}
    vec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{Vector{T}}

    big_Gmat::Matrix{T}
    big_Zmat::Vector{Matrix{T}}
    d2zdρ2G::Matrix{T}
    d2zdρ2Z::Vector{Matrix{T}}

    function EpiRenyiTri{T,R}(
        α::T,
        Gkraus::Vector{<:AbstractMatrix},
        Zkraus::Vector{<:AbstractMatrix},
        dim::Int;
        S::Union{AbstractMatrix,UniformScaling},
        blocks::Vector{<:AbstractVector} = [1:size(Zkraus[1], 1)],
        use_dual::Bool = false
    ) where {T<:Real,R<:RealOrComplex{T}}
        @assert dim > 1
        cone = new{T,R}()
        cone.use_dual_barrier = use_dual
        cone.blocks = blocks
        cone.nblocks = length(blocks)

        cone.α = α
        cone.α2 = (1 - α) / 2α
        cone.sα = α < 1 ? -1 : 1
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.ρ_dim = dim - 1
        cone.d = size(Gkraus[1], 2)
        cone.Gd = size(Gkraus[1], 1)
        cone.Zd = length.(blocks)
        cone.ZD = sum(cone.Zd)
        cone.Gρ_dim = Cones.svec_length(R, cone.Gd)
        cone.Zρ_dim = Cones.svec_length.(Ref(R), cone.Zd)
        cone.S = S

        Gkraus = [R.(Gk) for Gk ∈ Gkraus]
        Zkraus = [R.(Zk) for Zk ∈ Zkraus]
        cone.Gk = Gkraus
        cone.Zkbig = Zkraus
        cone.Zk = [filter!(!iszero, [Zk[blocks[i], :] for Zk ∈ Zkraus]) for i ∈ 1:cone.nblocks]
        cone.is_G_identity = (cone.Gk == [I(cone.d)])
        cone.is_S_identity = ((cone.S == [I(cone.Gd)]) || cone.S == I)
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

use_dder3(cone::EpiRenyiTri) = true

function reset_data(cone::EpiRenyiTri)
    return (
        cone.feas_updated =
            cone.grad_updated =
                cone.hess_updated =
                    cone.hess_aux_updated =
                        cone.inv_hess_updated =
                            cone.hess_fact_updated = cone.dder3_aux_updated = cone.inv_hess_aux_updated = false
    )
end

use_sqrt_hess_oracles(::Int, cone::EpiRenyiTri) = false

function setup_extra_data!(cone::EpiRenyiTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    d = cone.d
    Gd = cone.Gd
    Zd = cone.Zd
    ZD = cone.ZD
    ρ_dim = cone.ρ_dim
    Gρ_dim = cone.Gρ_dim
    Zρ_dim = cone.Zρ_dim

    cone.rt2 = sqrt(T(2))
    cone.ρ_idxs = 2:(ρ_dim+1)
    cone.ρ = zeros(R, d, d)
    cone.Gρ = zeros(R, Gd, Gd)
    cone.Gρroot = zeros(R, Gd, Gd)
    cone.Zρ = [zeros(R, s, s) for s ∈ Zd]
    cone.Zρα = [zeros(R, s, s) for s ∈ Zd]
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
    cone.Zρα_λ = [zeros(T, s) for s ∈ Zd]

    cone.Hρ = zeros(T, ρ_dim, ρ_dim)

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
    cone.ZGmat = zeros(R, ZD, Gd)
    cone.ZGmat2 = zeros(R, ZD, Gd)
    cone.ZG = zeros(R, ZD, Gd)
    cone.ZS = zeros(R, ZD, Gd)

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

get_nu(cone::EpiRenyiTri) = cone.d + 1

function set_initial_point!(arr::AbstractVector{T}, cone::EpiRenyiTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    d = cone.d
    blocks = cone.blocks

    γ = sqrt(T(d + 3) / (2d + 2) - 0.5 * cone.sα * sqrt(1 + T(4) / (d + 1)^2))

    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    for i ∈ 1:d
        arr[1+k] = γ
        k += incr * i + 1
    end
    @views ρ_vec = arr[cone.ρ_idxs]
    svec_to_smat!(cone.ρ, ρ_vec, cone.rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gρmat)
    end
    applykraus!.(cone.Zρ, cone.Zk, Ref(Hermitian(cone.ρ)), cone.Zρmat)
    cone.Gρ_fact = eigen(Hermitian(cone.Gρ))
    cone.Zρ_fact = eigen.(Hermitian.(cone.Zρ))
    Gρ_λ, Gρ_U = cone.Gρ_fact
    cone.Gρroot = Gρ_U * Diagonal(sqrt.(Gρ_λ)) * Gρ_U'
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    for i ∈ eachindex(Zρ_λ)
        cone.Zρα_λ[i] .= Zρ_λ[i] .^ cone.α2
    end
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    spectral_outer!.(cone.Zρα, Zρ_U, cone.Zρα_λ, cone.Zmat)
    if cone.is_S_identity
        for i ∈ eachindex(blocks), j ∈ eachindex(blocks)
            @views mul!(cone.ZG[blocks[i], :], cone.Zρα[i], cone.Gρroot[blocks[i], :])
        end
    else
        for i ∈ eachindex(blocks)
            @views mul!(cone.ZS[blocks[i], :], cone.Zρα[i], cone.S[blocks[i], :])
        end
        mul!(cone.ZG, cone.ZS, cone.Gρroot)
    end
    renyi = mapreduce(x -> x^(2 * cone.α), +, svdvals(cone.ZG))
    arr[1] = 0.5 * (cone.sα * renyi + sqrt(4 + renyi^2))
    return arr
end

function update_feas(cone::EpiRenyiTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.feas_updated
    @views ρ_vec = cone.point[cone.ρ_idxs]
    blocks = cone.blocks

    cone.is_feas = false

    svec_to_smat!(cone.ρ, ρ_vec, cone.rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gρmat)
    end
    applykraus!.(cone.Zρ, cone.Zk, Ref(Hermitian(cone.ρ)), cone.Zρmat)

    if isposdef(Hermitian(cone.ρ))
        cone.Gρ_fact = eigen(Hermitian(cone.Gρ))
        cone.Zρ_fact = eigen.(Hermitian.(cone.Zρ))
        if isposdef(cone.Gρ_fact) && all(isposdef.(cone.Zρ_fact)) #necessary because of numerical error
            Gρ_λ, Gρ_U = cone.Gρ_fact
            cone.Gρroot = Gρ_U * Diagonal(sqrt.(Gρ_λ)) * Gρ_U'
            Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
            for i ∈ eachindex(Zρ_λ)
                cone.Zρα_λ[i] .= Zρ_λ[i] .^ cone.α2
            end
            Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
            spectral_outer!.(cone.Zρα, Zρ_U, cone.Zρα_λ, cone.Zmat)
            if cone.is_S_identity
                for i ∈ eachindex(blocks)
                    @views mul!(cone.ZG[blocks[i], :], cone.Zρα[i], cone.Gρroot[blocks[i], :])
                end
            else
                for i ∈ eachindex(blocks)
                    @views mul!(cone.ZS[blocks[i], :], cone.Zρα[i], cone.S[blocks[i], :])
                end
                mul!(cone.ZG, cone.ZS, cone.Gρroot)
            end
            cone.ZG_fact = svd(cone.ZG)
            renyi = mapreduce(x -> x^(2 * cone.α), +, cone.ZG_fact.S)
            cone.z = cone.point[1] - cone.sα * renyi
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiRenyiTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert cone.is_feas
    blocks = cone.blocks

    zi = inv(cone.z)
    cone.grad[1] = -zi

    ## G part of gradient
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            @views mul!(cone.Gmat[blocks[i], :], cone.Zρα[i], cone.ZG_fact.U[blocks[i], :])
        end
    else
        mul!(cone.Gmat, cone.ZS', cone.ZG_fact.U)
    end
    mul!(cone.Gmat2, cone.Gmat, Diagonal(cone.ZG_fact.S .^ (cone.α - 1)))
    mul!(cone.Gmat3, cone.Gmat2, cone.Gmat2')
    if cone.is_G_identity
        cone.mat .= cone.α * cone.Gmat3
    else
        applykraus_adj!(cone.mat, cone.Gk, Hermitian(cone.Gmat3), cone.Gρmat)
        cone.mat .*= cone.α
    end

    ## Z part of gradient
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            @views mul!(cone.ZGmat2[blocks[i], :], Zρ_U[i]', cone.Gρroot[blocks[i], :])
        end
    else
        for i ∈ eachindex(blocks)
            @views mul!(cone.ZGmat[blocks[i], :], Zρ_U[i]', cone.S[blocks[i], :])
        end
        mul!(cone.ZGmat2, cone.ZGmat, cone.Gρroot)
    end
    mul!(cone.Gmat, cone.ZG_fact.V, Diagonal(cone.ZG_fact.S .^ (cone.α - 1)))
    mul!(cone.ZGmat, cone.ZGmat2, cone.Gmat)
    for i ∈ eachindex(blocks)
        @views mul!(cone.Zmat[i], cone.ZGmat[blocks[i], :], cone.ZGmat[blocks[i], :]', cone.α, false)
    end
    Zρα2_λ = [v .^ (2cone.α2) for v ∈ Zρ_λ]
    dZρα_λ = [2cone.α2 * (v .^ (2cone.α2 - 1)) for v ∈ Zρ_λ]
    Δ2generic!.(cone.Δ2Z, Zρ_λ, Zρα2_λ, dZρα_λ)   #Γ(Λ_Z)

    for i ∈ eachindex(blocks)
        cone.Zmat2[i] .= cone.Δ2Z[i] .* cone.Zmat[i]
    end
    spectral_outer!.(cone.Zmat3, Zρ_U, Hermitian.(cone.Zmat2), cone.Zmat)
    for i ∈ eachindex(blocks)
        applykraus_adj!(cone.mat2, cone.Zk[i], Hermitian(cone.Zmat3[i]), cone.Zρmat[1])
        cone.mat .+= cone.mat2
    end
    smat_to_svec!(cone.dzdρ, cone.mat, cone.rt2)
    cone.dzdρ .*= -cone.sα

    @. @views cone.grad[cone.ρ_idxs] = -zi * cone.dzdρ

    ## logdet part of gradient
    cone.ρ_fact = cone.is_G_identity ? cone.Gρ_fact : eigen(Hermitian(cone.ρ))
    ρ_λ, ρ_U = cone.ρ_fact
    cone.ρ_λ_inv .= inv.(ρ_λ)
    spectral_outer!(cone.ρ_inv, ρ_U, cone.ρ_λ_inv, cone.mat)
    smat_to_svec!(cone.vec, cone.ρ_inv, cone.rt2)
    @views cone.grad[cone.ρ_idxs] .-= cone.vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiRenyiTri)
    @assert cone.grad_updated

    Δ2!(cone.Δ2G, cone.Gρ_fact.values, cone.Gρ_λ_log)   #Γ(Λ_G)

    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Δ2!.(cone.Δ2Z, Zρ_λ, cone.Zρ_λ_log)   #Γ(Λ_Z)

    return cone.hess_aux_updated = true
end

function d2zdρ2!(
    d2zdρ2vec::AbstractVector{T},
    ρ_arr_mat::AbstractMatrix{R},
    cone::EpiRenyiTri{T,R}
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
function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiRenyiTri{T,R}
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
    @inbounds for i ∈ 1:size(arr, 2)
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

function update_inv_hess_auxold(cone::EpiRenyiTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
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

    @. cone.Hρ -= zi * d2zdρ2 # - 1/u ∇²ᵨᵨu + (ρ⁻¹) ̅ ⊗ρ⁻¹

    cone.Hρ_fact = Hypatia.posdef_fact!(Symmetric(cone.Hρ))
    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess_aux(cone::EpiRenyiTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.inv_hess_aux_updated
    @assert cone.grad_updated
    @assert cone.hess_aux_updated

    Gk = cone.Gk
    Zk = cone.Zk
    rt2 = cone.rt2
    dzdρ = cone.dzdρ
    d2zdρ2 = cone.d2zdρ2
    Gρ_U = cone.Gρ_fact.vectors
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]

    if cone.are_blocks_small #for small blocks it's more efficient to compute the little pieces with hessian_spectral_function! and later expand them into d2zdρ2
        d2zdρ2G = cone.d2zdρ2G
        d2zdρ2Z = cone.d2zdρ2Z

        eig_dot_kron!(d2zdρ2G, cone.Δ2G, Gρ_U, cone.Gmat, cone.Gmat2, cone.Gmat3, rt2)
        if cone.is_G_identity
            d2zdρ2 .= -1 .* d2zdρ2G
        else
            mul!(cone.big_Gmat, cone.Gadj, d2zdρ2G)
            mul!(d2zdρ2, cone.big_Gmat, cone.G, T(-1), false)
        end

        eig_dot_kron!.(d2zdρ2Z, cone.Δ2Z, Zρ_U, cone.Zmat, cone.Zmat2, cone.Zmat3, Ref(rt2))
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
        hessian_spectral_function!(d2zdρ2, cone.Δ2G, Gρ_U_adj_Gk, cone.Gmat2, cone.Gmat3, cone.Gρmat, cone.mat, rt2)
        d2zdρ2 .*= -1

        if all(length.(Zk) .== 1)
            Zρ_U_adj_Zk = [Zρ_U[i]' * Zk[i][1] for i ∈ 1:cone.nblocks]
        else
            Zρ_U_adj_Zk = [[Zρ_U[i]' * Zk[i][j] for j ∈ 1:length(Zk[i])] for i ∈ 1:cone.nblocks]
        end
        for i ∈ 1:cone.nblocks
            hessian_spectral_function!(
                cone.big_mat,
                cone.Δ2Z[i],
                Zρ_U_adj_Zk[i],
                cone.Zmat2[i],
                cone.Zmat3[i],
                cone.Zρmat[i],
                cone.mat,
                rt2
            )
            d2zdρ2 .+= cone.big_mat
        end
    end

    symm_kron!(cone.Hρ, cone.ρ_inv, rt2) # (ρ⁻¹) ̅ ⊗ρ⁻¹
    @. cone.Hρ -= inv(cone.z) * d2zdρ2 # - 1/u ∇²ᵨᵨu + (ρ⁻¹) ̅ ⊗ρ⁻¹

    cone.Hρ_fact = Hypatia.posdef_fact!(Symmetric(cone.Hρ))
    cone.inv_hess_aux_updated = true
    return
end

#uses the decomposition from appendix B.2 of arXiv:2407.00241
function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiRenyiTri{T,R}
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

function update_dder3_aux(cone::EpiRenyiTri)
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
    cone::EpiRenyiTri{T,R}
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

function dder3(cone::EpiRenyiTri{T,R}, dir::AbstractVector{T}) where {T<:Real,R<:RealOrComplex{T}}
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
