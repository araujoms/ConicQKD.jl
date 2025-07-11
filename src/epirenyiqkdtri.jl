mutable struct EpiRenyiQKDTri{T<:Real,R<:RealOrComplex{T}} <: Cone{T}
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
    sqrtGρ::Matrix{R}
    Zρ::Vector{Matrix{R}}
    hZρ::Vector{Matrix{R}}
    ShZρ::Matrix{R}
    sqrtShZρ::Matrix{R}
    invsqrtShZρ::Matrix{R}
    G::Matrix{T}
    S::Union{Matrix{R},UniformScaling{Bool}} #FIXME abstract type
    Z::Vector{Matrix{T}}
    Gk::Vector{Matrix{R}}
    Zk::Vector{Vector{Matrix{R}}}
    Zkbig::Vector{Matrix{R}}
    Gadj::Matrix{T}
    Zadj::Vector{Matrix{T}}
    ρ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    Gρ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    Zρ_fact::Vector{Eigen{R,T,Matrix{R},Vector{T}}}
    ZG_fact::SVD{R,T,Matrix{R},Vector{T}}
    ρ_inv::Matrix{R}
    ρ_λ_inv::Vector{T}
    Gρ_λ_log::Vector{T}
    Zρ_λ_log::Vector{Vector{T}}
    Zρα_λ::Vector{Vector{T}}
    hZρ_λ::Vector{Vector{T}}
    z::T
    Δ2G::Matrix{T}
    Δ2_h_Zρ::Vector{Matrix{T}}
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

    function EpiRenyiQKDTri{T,R}(
        α::T,
        Gkraus::Vector{<:AbstractMatrix},
        Zkraus::Vector{<:AbstractMatrix},
        dim::Int;
        S::Union{AbstractMatrix,UniformScaling},
        blocks::Vector{UnitRange{Int}} = [1:size(Zkraus[1], 1)],
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

use_dder3(cone::EpiRenyiQKDTri) = false

function reset_data(cone::EpiRenyiQKDTri)
    return (
        cone.feas_updated =
            cone.grad_updated =
                cone.hess_updated =
                    cone.hess_aux_updated =
                        cone.inv_hess_updated =
                            cone.hess_fact_updated = cone.dder3_aux_updated = cone.inv_hess_aux_updated = false
    )
end

function setup_extra_data!(cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
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
    cone.sqrtGρ = zeros(R, Gd, Gd)
    cone.Zρ = [zeros(R, s, s) for s ∈ Zd]
    cone.hZρ = [zeros(R, s, s) for s ∈ Zd]
    cone.ShZρ = zeros(R, Gd, Gd)
    cone.sqrtShZρ = zeros(R, Gd, Gd)
    cone.invsqrtShZρ = zeros(R, Gd, Gd)
    cone.ρ_inv = zeros(R, d, d)
    cone.dzdρ = zeros(T, ρ_dim)
    cone.d2zdρ2vec = zeros(T, ρ_dim)
    cone.Δ2G = zeros(T, Gd, Gd)
    cone.Δ2_h_Zρ = [zeros(T, s, s) for s ∈ Zd]
    cone.Δ3G = zeros(T, Gd, Gd, Gd)
    cone.Δ3Z = [zeros(T, s, s, s) for s ∈ Zd]
    cone.d2zdρ2 = zeros(T, ρ_dim, ρ_dim)
    cone.ρ_λ_inv = zeros(T, d)
    cone.Gρ_λ_log = zeros(T, Gd)
    cone.Zρ_λ_log = [zeros(T, s) for s ∈ Zd]
    cone.Zρα_λ = [zeros(T, s) for s ∈ Zd]
    cone.hZρ_λ = [zeros(T, s) for s ∈ Zd]

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
    cone.ZG = zeros(R, Gd, Gd)
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

get_nu(cone::EpiRenyiQKDTri) = cone.d + 1

function set_initial_point!(arr::AbstractVector{T}, cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    d = cone.d
    blocks = cone.blocks
    h(x) = x^((1 - cone.α) / cone.α)

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
    cone.sqrtGρ = Gρ_U * Diagonal(sqrt.(Gρ_λ)) * Gρ_U'

    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    for i ∈ eachindex(Zρ_λ)
        cone.hZρ_λ[i] .= h.(Zρ_λ[i])
    end
    spectral_outer!.(cone.hZρ, Zρ_U, cone.hZρ_λ, cone.Zmat)
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            @views cone.ShZρ[blocks[i], blocks[i]] .= cone.hZρ[i]
            @views spectral_outer!(cone.sqrtShZρ[blocks[i], blocks[i]], Zρ_U[i], sqrt.(cone.hZρ_λ[i]), cone.Zmat[i])
        end
    else
        for i ∈ eachindex(blocks)
            @views mul!(cone.ZS[blocks[i], :], cone.hZρ[i], cone.S[blocks[i], :])
        end
        mul!(cone.ShZρ, cone.S', cone.ZS)
        cone.sqrtShZρ .= sqrt(Hermitian(cone.ShZρ))
    end
    mul!(cone.ZG, cone.sqrtShZρ, cone.sqrtGρ)
    cone.ZG_fact = svd(cone.ZG)
    renyi = mapreduce(x -> x^(2 * cone.α), +, svdvals(cone.ZG))

    arr[1] = 0.5 * (cone.sα * renyi + sqrt(4 + renyi^2))
    return arr
end

function update_feas(cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.feas_updated
    @views ρ_vec = cone.point[cone.ρ_idxs]
    blocks = cone.blocks
    h(x) = x^((1 - cone.α) / cone.α)

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
        cone.ρ_fact = cone.is_G_identity ? cone.Gρ_fact : eigen(Hermitian(cone.ρ))
        cone.Zρ_fact = eigen.(Hermitian.(cone.Zρ))
        if isposdef(cone.ρ_fact) && isposdef(cone.Gρ_fact) && all(isposdef.(cone.Zρ_fact)) #necessary because of numerical error
            Gρ_λ, Gρ_U = cone.Gρ_fact
            cone.sqrtGρ = Gρ_U * Diagonal(sqrt.(Gρ_λ)) * Gρ_U'
            Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
            Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
            for i ∈ eachindex(Zρ_λ)
                cone.hZρ_λ[i] .= h.(Zρ_λ[i])
            end
            spectral_outer!.(cone.hZρ, Zρ_U, cone.hZρ_λ, cone.Zmat)
            if cone.is_S_identity
                for i ∈ eachindex(blocks)
                    @views cone.ShZρ[blocks[i], blocks[i]] .= cone.hZρ[i]
                    @views spectral_outer!(
                        cone.sqrtShZρ[blocks[i], blocks[i]],
                        Zρ_U[i],
                        sqrt.(cone.hZρ_λ[i]),
                        cone.Zmat[i]
                    )
                    @views spectral_outer!(
                        cone.invsqrtShZρ[blocks[i], blocks[i]],
                        Zρ_U[i],
                        inv.(sqrt.(cone.hZρ_λ[i])),
                        cone.Zmat[i]
                    )
                end
            else
                for i ∈ eachindex(blocks)
                    @views mul!(cone.ZS[blocks[i], :], cone.hZρ[i], cone.S[blocks[i], :])
                end
                mul!(cone.ShZρ, cone.S', cone.ZS)
                cone.sqrtShZρ .= sqrt(Hermitian(cone.ShZρ))
                cone.invsqrtShZρ .= inv(Hermitian(cone.sqrtShZρ))
            end
            mul!(cone.ZG, cone.sqrtShZρ, cone.sqrtGρ)
            cone.ZG_fact = svd(cone.ZG)
            renyi = mapreduce(x -> x^(2 * cone.α), +, cone.ZG_fact.S)
            cone.z = cone.point[1] - cone.sα * renyi
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert cone.is_feas
    blocks = cone.blocks
    dh(x) = ((1 - cone.α) / cone.α) * x^((1 - cone.α) / cone.α - 1)

    zi = inv(cone.z)
    cone.grad[1] = -zi

    ## G part of gradient
    mul!(cone.Gmat, cone.sqrtShZρ, cone.ZG_fact.U)
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
            @views mul!(cone.ZGmat2[blocks[i], :], Zρ_U[i]', cone.sqrtGρ[blocks[i], :])
        end
    else
        for i ∈ eachindex(blocks)
            @views mul!(cone.ZGmat[blocks[i], :], Zρ_U[i]', cone.S[blocks[i], :])
        end
        mul!(cone.ZGmat2, cone.ZGmat, cone.sqrtGρ)
    end
    mul!(cone.Gmat, cone.ZG_fact.V, Diagonal(cone.ZG_fact.S .^ (cone.α - 1)))
    mul!(cone.ZGmat, cone.ZGmat2, cone.Gmat)
    for i ∈ eachindex(blocks)
        @views mul!(cone.Zmat[i], cone.ZGmat[blocks[i], :], cone.ZGmat[blocks[i], :]', cone.α, false)
    end
    dhZρ_λ = [dh.(v) for v ∈ Zρ_λ]
    Δ2generic!.(cone.Δ2_h_Zρ, Zρ_λ, cone.hZρ_λ, dhZρ_λ)   #Γ(Λ_Z)

    for i ∈ eachindex(blocks)
        cone.Zmat2[i] .= cone.Δ2_h_Zρ[i] .* cone.Zmat[i]
    end
    spectral_outer!.(cone.Zmat3, Zρ_U, Hermitian.(cone.Zmat2), cone.Zmat)
    for i ∈ eachindex(blocks)
        applykraus_adj!(cone.mat2, cone.Zk[i], Hermitian(cone.Zmat3[i]), cone.Zρmat[1])
        cone.mat .+= cone.mat2
    end
    smat_to_svec!(cone.dzdρ, cone.mat, cone.rt2)

    @. @views cone.grad[cone.ρ_idxs] = zi * cone.sα * cone.dzdρ

    ## logdet part of gradient
    ρ_λ, ρ_U = cone.ρ_fact
    cone.ρ_λ_inv .= inv.(ρ_λ)
    spectral_outer!(cone.ρ_inv, ρ_U, cone.ρ_λ_inv, cone.mat)
    smat_to_svec!(cone.vec, cone.ρ_inv, cone.rt2)
    @views cone.grad[cone.ρ_idxs] .-= cone.vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiRenyiQKDTri)
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    zi = inv(cone.z)
    H[1, 1] = abs2(zi) #∇hh = 1/z^2
    @views @. H[1, cone.ρ_idxs] = -abs2(zi) * cone.sα * cone.dzdρ #∇hρ = sα/z^2 * ∇ρ Ψ
    @views Hρ = H[cone.ρ_idxs, cone.ρ_idxs]
    @views mul!(Hρ, cone.dzdρ, cone.dzdρ', abs2(zi), false) #∇ρρ = 1/z^2 * (∇ρ Ψ) * (∇ρ Ψ)'

    α = cone.α
    g(x) = x^α
    dg(x) = α * x^(α - 1)
    d2g(x) = α * (α - 1) * x^(α - 2)
    g̃(x) = α * x^α
    dg̃(x) = α^2 * x^(α - 1)

    h(x) = x^(1 / α - 1)
    rooth(x) = x^((1 - α) / (2α))
    dh(x) = (1 / α - 1) * x^(1 / α - 2)
    d2h(x) = (1 / α - 1) * (1 / α - 2) * x^(1 / α - 3)

    Gρ = cone.Gρ
    #Zρ = sum(K * cone.ρ * K' for K ∈ cone.Zkbig)
    Zρ = zeros(eltype(Gρ), cone.ZD, cone.ZD)
    for i ∈ eachindex(cone.blocks)
        @views Zρ[cone.blocks[i], cone.blocks[i]] .= Zρ[i]
    end
    S = cone.S
    Gmatrix = sum(skron.(cone.Gk))
    Zmatrix = sum(skron.(cone.Zkbig))
    rootGρ = cone.sqrtGρ
    ZSρ = cone.ShZρ
    rootZSρ = cone.sqrtShZρ
    invrootZSρ = cone.invsqrtShZρ
    ZGZ = Hermitian(rootZSρ * Gρ * rootZSρ)
    GZG = Hermitian(rootGρ * ZSρ * rootGρ)
    dρvec = cone.ρ_dim
    d2zdρ2 = cone.d2zdρ2

    #GG
    vecZZ = skron(rootZSρ)
    λ_ZGZ, U_ZGZ = eigen(ZGZ)
    Δ2_dg_ZGZ = Δ2generic(λ_ZGZ, dg.(λ_ZGZ), d2g.(λ_ZGZ))
    dsfdg_ZGZ = d_spectral(Δ2_dg_ZGZ, Matrix(U_ZGZ'))
    d2zdρ2 .= Gmatrix' * vecZZ * dsfdg_ZGZ * vecZZ * Gmatrix

    #ZG
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    #λz, Uz = eigen(Hermitian(Zρ))
    λz = reduce(vcat, Zρ_λ)
    Uz = zeros(eltype(Gρ), cone.ZD, cone.ZD)
    for i ∈ eachindex(cone.blocks)
        @views Uz[cone.blocks[i], cone.blocks[i]] .= Zρ_U[i]
    end
    Δ2z = Δ2generic(λz, h.(λz), dh.(λz))
    dsfh = d_spectral(Δ2z, Matrix(Uz'))
    vecZSSZ = skron(invrootZSρ * S')
    Δ2_g̃_ZGZ = Δ2generic(λ_ZGZ, g̃.(λ_ZGZ), dg̃.(λ_ZGZ))
    dsfg̃ = d_spectral(Δ2_g̃_ZGZ, Matrix(U_ZGZ'))
    HGZ = Gmatrix' * vecZZ * dsfg̃ * vecZSSZ * dsfh * Zmatrix
    d2zdρ2 .+= HGZ + HGZ'

    #ZZ
    vecGSSG = skron(rootGρ * S')
    λ_GZG, U_GZG = eigen(GZG)
    Δ2_dg_GZG = Δ2generic(λ_GZG, dg.(λ_GZG), d2g.(λ_GZG))
    dsfdg_GZG = d_spectral(Δ2_dg_GZG, Matrix(U_GZG'))
    first_term = dsfh * vecGSSG' * dsfdg_GZG * vecGSSG * dsfh

    W = S * rootGρ * dg(GZG) * rootGρ * S'
    Δ3z = Δ3generic(Δ2z, λz, d2h.(λz))
    second_term = d2_spectral(Δ3z, Uz, W)
    d2zdρ2 .+= Zmatrix' * (first_term + second_term) * Zmatrix

    @. Hρ += zi * cone.sα * d2zdρ2 #∇ρρ += sα/z ∇ρρ Ψ
    #logdet part
    symm_kron!(cone.big_mat, cone.ρ_inv, cone.rt2) # ∇ρρ += skron(ρ⁻¹)
    Hρ .+= cone.big_mat
    cone.hess_updated = true
    return cone.hess
end
