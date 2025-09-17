mutable struct EpiRenyiQKDTri{T<:Real,R<:RealOrComplex{T}} <: Cone{T}
    α::T
    sα::Int
    use_dual_barrier::Bool
    dim::Int
    ρd::Int
    σd::Int
    Gd::Int
    Zd::Vector{Int}
    is_complex::Bool
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
    σ_dim::Int
    Gρ_dim::Int
    Zσ_dim::Vector{Int}
    ρ_idxs::UnitRange{Int}
    σ_idxs::UnitRange{Int}
    ρ::Matrix{R}
    σ::Matrix{R}
    Gρ::Matrix{R}
    sqrtGρ::Matrix{R}
    invsqrtGρ::Matrix{R}
    Zσ::Vector{Matrix{R}}
    hZσ::Vector{Matrix{R}}
    ShZσ::Matrix{R}
    sqrtShZσ::Matrix{R}
    invsqrtShZσ::Matrix{R}
    G::Matrix{T}
    S::Matrix{R}
    Z::Vector{Matrix{T}}
    Gk::Vector{Matrix{R}}
    Zk::Vector{Vector{Matrix{R}}}
    Gadj::Matrix{T}
    Zadj::Vector{Matrix{T}}
    ρ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    σ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    Gρ_fact::Eigen{R,T,Matrix{R},Vector{T}}
    Zσ_fact::Vector{Eigen{R,T,Matrix{R},Vector{T}}}
    ZG_fact::SVD{R,T,Matrix{R},Vector{T}}
    ρ_inv::Matrix{R}
    σ_inv::Matrix{R}
    ρ_λ_inv::Vector{T}
    σ_λ_inv::Vector{T}
    Gρ_λ_log::Vector{T}
    Zσ_λ_log::Vector{Vector{T}}
    hZσ_λ::Vector{Vector{T}}
    z::T
    Δ2_dg_ZGZ::Matrix{T}
    Δ3_dg_ZGZ::Array{T,3}
    Δ2_g̃_ZGZ::Matrix{T}
    Δ3_g̃_ZGZ::Array{T,3}
    Δ2_h_Zσ::Vector{Matrix{T}}
    Δ3_h_Zσ::Vector{Array{T,3}}
    Δ3_h_ZσW̃::Vector{Array{R,3}}
    Δ4_ij_h_Zσ::Vector{Matrix{T}}
    dΨdρ::Vector{T}
    dΨdσ::Vector{T}
    d2Ψdρ2vec::Vector{T}
    d2Ψdσ2vec::Vector{T}
    DhZmeat::Vector{Matrix{R}}
    ds_g̃_ZGZ::Matrix{T} #TODO check if it's being reused in dder3
    ds_h_Zσ::Vector{Matrix{T}}

    ZG::Matrix{R}
    #variables below are just scratch space
    ρmat::Matrix{R}
    ρmat2::Matrix{R}
    ρmat3::Matrix{R}
    σmat::Matrix{R}
    σmat2::Matrix{R}
    σmat3::Matrix{R}
    Gmat::Matrix{R}
    Gmat2::Matrix{R}
    Gmat3::Matrix{R}
    Gmat4::Matrix{R}
    Gmat5::Matrix{R}
    Gmat6::Matrix{R}
    Gρmat::Matrix{R}
    Gρmatvec::Vector{Matrix{R}}
    Zmat::Vector{Matrix{R}}
    Zmat2::Vector{Matrix{R}}
    Zmat3::Vector{Matrix{R}}
    Zmat4::Vector{Matrix{R}}
    Zσmat::Vector{Matrix{R}}
    ZGmat::Vector{Matrix{R}}
    ZGmat2::Vector{Matrix{R}}

    ρvec::Vector{T}
    σvec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{Vector{T}}

    big_ρmat::Matrix{T}
    big_σmat::Matrix{T}
    big_σρmat::Matrix{T}
    big_Gmat::Matrix{T}
    big_Gρmat::Matrix{T}
    big_σGmat::Matrix{T}
    big_σGmat2::Matrix{T}
    big_Zmat::Vector{Matrix{T}}
    big_ZGmat::Vector{Matrix{T}}
    big_σZmat::Vector{Matrix{T}}

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
        cone.sα = α < 1 ? -1 : 1
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.ρd = size(Gkraus[1], 2)
        cone.σd = size(Zkraus[1], 2)
        cone.Gd = size(Gkraus[1], 1)
        cone.Zd = length.(blocks)
        cone.ρ_dim = Cones.svec_length(R, cone.ρd)
        cone.σ_dim = Cones.svec_length(R, cone.σd)
        cone.Gρ_dim = Cones.svec_length(R, cone.Gd)
        cone.Zσ_dim = Cones.svec_length.(Ref(R), cone.Zd)

        Gkraus = [R.(Gk) for Gk ∈ Gkraus]
        Zkraus = [R.(Zk) for Zk ∈ Zkraus]
        cone.Gk = Gkraus
        cone.Zk = [filter!(!iszero, [Zk[blocks[i], :] for Zk ∈ Zkraus]) for i ∈ 1:cone.nblocks]
        cone.is_G_identity = (cone.Gk == [I(cone.ρd)])
        cone.is_S_identity = (S == I)
        if cone.is_S_identity
            cone.S = fill(R(1), 1, 1) #the goal is to error if S is used
        else
            cone.S = convert(Matrix{R}, S)
        end
        cone.G = kraus2matrix(Gkraus)
        cone.Z = [kraus2matrix([Zk[blocks[i], :] for Zk ∈ Zkraus]) for i ∈ 1:length(blocks)]
        cone.Gadj = Matrix(cone.G')
        cone.Zadj = Matrix.(adjoint.(cone.Z))
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
    ρd = cone.ρd
    σd = cone.σd
    Gd = cone.Gd
    Zd = cone.Zd
    ρ_dim = cone.ρ_dim
    σ_dim = cone.σ_dim
    Gρ_dim = cone.Gρ_dim
    Zσ_dim = cone.Zσ_dim

    cone.rt2 = sqrt(T(2))
    cone.ρ_idxs = 2:(ρ_dim+1)
    cone.σ_idxs = (ρ_dim+2):(ρ_dim+σ_dim+1)
    cone.ρ = zeros(R, ρd, ρd)
    cone.σ = zeros(R, σd, σd)
    cone.Gρ = zeros(R, Gd, Gd)
    cone.sqrtGρ = zeros(R, Gd, Gd)
    cone.invsqrtGρ = zeros(R, Gd, Gd)
    cone.Zσ = [zeros(R, s, s) for s ∈ Zd]
    cone.hZσ = [zeros(R, s, s) for s ∈ Zd]
    cone.ShZσ = zeros(R, Gd, Gd)
    cone.sqrtShZσ = zeros(R, Gd, Gd)
    cone.invsqrtShZσ = zeros(R, Gd, Gd)
    cone.ρ_inv = zeros(R, ρd, ρd)
    cone.σ_inv = zeros(R, σd, σd)
    cone.dΨdρ = zeros(T, ρ_dim)
    cone.dΨdσ = zeros(T, σ_dim)
    cone.d2Ψdρ2vec = zeros(T, ρ_dim)
    cone.d2Ψdσ2vec = zeros(T, σ_dim)
    cone.Δ2_dg_ZGZ = zeros(T, Gd, Gd)
    cone.Δ3_dg_ZGZ = zeros(T, Gd, Gd, Gd)
    cone.Δ2_g̃_ZGZ = zeros(T, Gd, Gd)
    cone.Δ3_g̃_ZGZ = zeros(T, Gd, Gd, Gd)
    cone.Δ2_h_Zσ = [zeros(T, s, s) for s ∈ Zd]
    cone.Δ3_h_Zσ = [zeros(T, s, s, s) for s ∈ Zd]
    cone.Δ3_h_ZσW̃ = [zeros(R, s, s, s) for s ∈ Zd]
    cone.Δ4_ij_h_Zσ = [zeros(T, s, s) for s ∈ Zd]
    cone.ρ_λ_inv = zeros(T, ρd)
    cone.σ_λ_inv = zeros(T, σd)
    cone.Gρ_λ_log = zeros(T, Gd)
    cone.Zσ_λ_log = [zeros(T, s) for s ∈ Zd]
    cone.hZσ_λ = [zeros(T, s) for s ∈ Zd]
    cone.DhZmeat = [zeros(R, s, s) for s ∈ Zd]

    cone.ρmat = zeros(R, ρd, ρd)
    cone.ρmat2 = zeros(R, ρd, ρd)
    cone.ρmat3 = zeros(R, ρd, ρd)
    cone.σmat = zeros(R, σd, σd)
    cone.σmat2 = zeros(R, σd, σd)
    cone.σmat3 = zeros(R, σd, σd)
    cone.Gmat = zeros(R, Gd, Gd)
    cone.Gmat2 = zeros(R, Gd, Gd)
    cone.Gmat3 = zeros(R, Gd, Gd)
    cone.Gmat4 = zeros(R, Gd, Gd)
    cone.Gmat5 = zeros(R, Gd, Gd)
    cone.Gmat6 = zeros(R, Gd, Gd)
    cone.Gρmat = zeros(R, Gd, ρd)
    cone.Gρmatvec = [zeros(R, Gd, ρd) for _ ∈ 1:length(cone.Gk)]
    cone.Zmat = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat2 = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat3 = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat4 = [zeros(R, s, s) for s ∈ Zd]
    cone.ZGmat = [zeros(R, s, Gd) for s ∈ Zd]
    cone.ZGmat2 = [zeros(R, s, Gd) for s ∈ Zd]
    cone.Zσmat = [zeros(R, s, σd) for s ∈ Zd]
    cone.ZG = zeros(R, Gd, Gd)

    cone.ρvec = zeros(T, ρ_dim)
    cone.σvec = zeros(T, σ_dim)
    cone.Gvec = zeros(T, Gρ_dim)
    cone.Zvec = [zeros(T, s) for s ∈ Zσ_dim]
    cone.ds_g̃_ZGZ = zeros(T, Gρ_dim, Gρ_dim)
    cone.ds_h_Zσ = [zeros(T, s, s) for s ∈ Zσ_dim]
    cone.big_ρmat = zeros(T, ρ_dim, ρ_dim)
    cone.big_σmat = zeros(T, σ_dim, σ_dim)
    cone.big_σρmat = zeros(T, σ_dim, ρ_dim)
    cone.big_Gmat = zeros(T, Gρ_dim, Gρ_dim)
    cone.big_Gρmat = zeros(T, Gρ_dim, ρ_dim)
    cone.big_σGmat = zeros(T, σ_dim, Gρ_dim)
    cone.big_σGmat2 = zeros(T, σ_dim, Gρ_dim)
    cone.big_Zmat = [zeros(T, s, s) for s ∈ Zσ_dim]
    cone.big_ZGmat = [zeros(T, s, Gρ_dim) for s ∈ Zσ_dim]
    cone.big_σZmat = [zeros(T, σ_dim, s) for s ∈ Zσ_dim]
    return
end

get_nu(cone::EpiRenyiQKDTri) = cone.ρd + cone.σd + 1

function _initial_γδ(α::T, d::Integer) where {T<:AbstractFloat}
    if α < 1
        γ = √(1 + α / d)
    else
        γ = (((2α - 1) / α)^(α - 1) / (d * α + α^2))^(1 / (2α))
    end
    tol = sqrt(eps(T))
    maxiter = 2ceil(log2(-log2(tol)))
    counter = 0
    while counter < maxiter
        counter += 1
        step = _newton_ratio(γ, α, d)
        γ -= step
        if abs(step) < tol
            break
        end
    end
    counter == maxiter && error("Failed to compute initial point.")
    δ = √((γ^2 - 1) * (1 - α) / α + 1)
    return γ, δ
end

function _newton_ratio(γ, α, d)
    γ2m1 = γ^2 - 1
    δ2 = 1 + γ2m1 * (1 - α) / α
    f = γ2m1^2 * γ^(-2α) * δ2^(α - 1) + d * α * γ2m1 - α^2
    df = γ2m1 * γ^(-2α - 1) * δ2^(α - 1) * (4γ^2 - 2α * γ2m1 - 2γ2m1 * γ^2 * (1 - α)^2 / (δ2 * α)) + 2d * α * γ
    return f / df
end

function set_initial_point!(arr::AbstractVector{T}, cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    blocks = cone.blocks
    ShZσ = cone.ShZσ
    sqrtShZσ = cone.sqrtShZσ
    invsqrtShZσ = cone.invsqrtShZσ
    hZσ = cone.hZσ
    hZσ_λ = cone.hZσ_λ
    Gmat = cone.Gmat
    Zmat = cone.Zmat

    h(x) = x^((1 - cone.α) / cone.α)

    γ, δ = _initial_γδ(cone.α, cone.ρd)

    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    for i ∈ 1:cone.ρd
        arr[1+k] = γ
        k += incr * i + 1
    end
    k = 1
    for i ∈ 1:cone.σd
        arr[1+cone.ρ_dim+k] = δ
        k += incr * i + 1
    end
    @views svec_to_smat!(cone.ρ, arr[cone.ρ_idxs], cone.rt2)
    @views svec_to_smat!(cone.σ, arr[cone.σ_idxs], cone.rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gρmat)
    end
    applykraus!.(cone.Zσ, cone.Zk, Ref(Hermitian(cone.σ)), cone.Zσmat)
    cone.Gρ_fact = eigen(Hermitian(cone.Gρ))
    cone.Zσ_fact = eigen.(Hermitian.(cone.Zσ))
    Gρ_λ, Gρ_U = cone.Gρ_fact
    mul!(Gmat, Gρ_U, Diagonal(fourthroot.(Gρ_λ)))
    mul!(cone.sqrtGρ, Gmat, Gmat')
    Zσ_λ = [fact.values for fact ∈ cone.Zσ_fact]
    Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]
    for i ∈ eachindex(Zσ_λ)
        hZσ_λ[i] .= h.(Zσ_λ[i])
    end
    spectral_outer!.(hZσ, Zσ_U, hZσ_λ, Zmat)
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            @views ShZσ[blocks[i], blocks[i]] .= hZσ[i]
            mul!(Zmat[i], Zσ_U[i], Diagonal(fourthroot.(hZσ_λ[i])))
            @views mul!(sqrtShZσ[blocks[i], blocks[i]], Zmat[i], Zmat[i]')
        end
    else
        fill!(ShZσ, 0)
        for i ∈ eachindex(blocks)
            @views spectral_outer!(Gmat, cone.S[blocks[i], :]', Hermitian(hZσ[i]), cone.ZGmat[i])
            ShZσ .+= Gmat
        end
        ShZσ_λ, ShZσ_U = eigen(Hermitian(ShZσ))
        mul!(Gmat, ShZσ_U, Diagonal(fourthroot.(ShZσ_λ)))
        mul!(sqrtShZσ, Gmat, Gmat')
    end
    mul!(cone.ZG, cone.sqrtShZσ, cone.sqrtGρ)
    renyi = mapreduce(x -> x^(2 * cone.α), +, svdvals(cone.ZG))

    arr[1] = (cone.sα * renyi + sqrt(4 + renyi^2)) / 2
    return arr
end

function update_feas(cone::EpiRenyiQKDTri)
    @assert !cone.feas_updated
    @views ρ_vec = cone.point[cone.ρ_idxs]
    @views σ_vec = cone.point[cone.σ_idxs]
    blocks = cone.blocks
    ShZσ = cone.ShZσ
    sqrtShZσ = cone.sqrtShZσ
    invsqrtShZσ = cone.invsqrtShZσ
    hZσ = cone.hZσ
    hZσ_λ = cone.hZσ_λ
    Gmat = cone.Gmat
    Zmat = cone.Zmat

    h(x) = x^((1 - cone.α) / cone.α)

    cone.is_feas = false

    svec_to_smat!(cone.ρ, ρ_vec, cone.rt2)
    svec_to_smat!(cone.σ, σ_vec, cone.rt2)
    if cone.is_G_identity
        cone.Gρ = cone.ρ
    else
        applykraus!(cone.Gρ, cone.Gk, Hermitian(cone.ρ), cone.Gρmat)
    end
    applykraus!.(cone.Zσ, cone.Zk, Ref(Hermitian(cone.σ)), cone.Zσmat)

    if isposdef(Hermitian(cone.ρ)) && isposdef(Hermitian(cone.σ))
        cone.ρ_fact = eigen(Hermitian(cone.ρ))
        cone.σ_fact = eigen(Hermitian(cone.σ))
        cone.Gρ_fact = cone.is_G_identity ? cone.ρ_fact : eigen(Hermitian(cone.Gρ))
        cone.Zσ_fact = eigen.(Hermitian.(cone.Zσ))
        if isposdef(cone.ρ_fact) && isposdef(cone.σ_fact) && isposdef(cone.Gρ_fact) && all(isposdef.(cone.Zσ_fact)) #necessary because of numerical error
            Gρ_λ, Gρ_U = cone.Gρ_fact
            mul!(Gmat, Gρ_U, Diagonal(fourthroot.(Gρ_λ)))
            mul!(cone.sqrtGρ, Gmat, Gmat')
            mul!(Gmat, Gρ_U, Diagonal(map(inv ∘ fourthroot, Gρ_λ)))
            mul!(cone.invsqrtGρ, Gmat, Gmat')

            Zσ_λ = [fact.values for fact ∈ cone.Zσ_fact]
            Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]
            for i ∈ eachindex(Zσ_λ)
                hZσ_λ[i] .= h.(Zσ_λ[i])
            end
            spectral_outer!.(hZσ, Zσ_U, hZσ_λ, Zmat)
            if cone.is_S_identity
                for i ∈ eachindex(blocks)
                    @views ShZσ[blocks[i], blocks[i]] .= hZσ[i]
                    mul!(Zmat[i], Zσ_U[i], Diagonal(fourthroot.(hZσ_λ[i])))
                    @views mul!(sqrtShZσ[blocks[i], blocks[i]], Zmat[i], Zmat[i]')
                    mul!(Zmat[i], Zσ_U[i], Diagonal(map(inv ∘ fourthroot, hZσ_λ[i])))
                    @views mul!(invsqrtShZσ[blocks[i], blocks[i]], Zmat[i], Zmat[i]')
                end
            else
                fill!(ShZσ, 0)
                for i ∈ eachindex(blocks)
                    @views spectral_outer!(Gmat, cone.S[blocks[i], :]', Hermitian(hZσ[i]), cone.ZGmat[i])
                    ShZσ .+= Gmat
                end
                ShZσ_λ, ShZσ_U = eigen(Hermitian(ShZσ))
                mul!(Gmat, ShZσ_U, Diagonal(fourthroot.(ShZσ_λ)))
                mul!(sqrtShZσ, Gmat, Gmat')
                mul!(Gmat, ShZσ_U, Diagonal(map(inv ∘ fourthroot, ShZσ_λ)))
                mul!(invsqrtShZσ, Gmat, Gmat')
            end
            mul!(cone.ZG, cone.sqrtShZσ, cone.sqrtGρ)
            cone.ZG_fact = svd(cone.ZG)
            renyi = mapreduce(x -> x^(2 * cone.α), +, cone.ZG_fact.S)
            cone.z = cone.point[1] - cone.sα * renyi
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiRenyiQKDTri)
    @assert cone.is_feas
    blocks = cone.blocks

    α = cone.α
    sqrtGρ = cone.sqrtGρ
    S = cone.S
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    ZGmat = cone.ZGmat
    ZGmat2 = cone.ZGmat2

    h(x) = x^(1 / α - 1)
    dh(x) = (1 / α - 1) * x^(1 / α - 2)

    zi = inv(cone.z)
    cone.grad[1] = -zi

    ## G part of gradient
    U_ZGZ = cone.ZG_fact.U
    mul!(Gmat, cone.sqrtShZσ, U_ZGZ)
    mul!(Gmat2, Gmat, Diagonal(cone.ZG_fact.S .^ (cone.α - 1)))
    mul!(Gmat3, Gmat2, Gmat2')
    if cone.is_G_identity
        cone.ρmat .= α .* Gmat3
    else
        applykraus_adj!(cone.ρmat, cone.Gk, Hermitian(Gmat3), cone.Gρmat)
        cone.ρmat .*= α
    end
    smat_to_svec!(cone.dΨdρ, cone.ρmat, cone.rt2)

    ## Z part of gradient
    U_GZG = cone.ZG_fact.V
    Zσ_λ = [fact.values for fact ∈ cone.Zσ_fact]
    Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]
    if cone.is_S_identity
        mul!(Gmat2, U_GZG, Diagonal(cone.ZG_fact.S .^ (α - 1)))
    else
        mul!(Gmat, U_GZG, Diagonal(cone.ZG_fact.S .^ (α - 1)))
        mul!(Gmat2, sqrtGρ, Gmat)
    end
    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], sqrtGρ[blocks[i], :], Gmat2)
        else
            @views mul!(ZGmat[i], S[blocks[i], :], Gmat2)
        end
        mul!(ZGmat2[i], Zσ_U[i]', ZGmat[i])
        mul!(cone.DhZmeat[i], ZGmat2[i], ZGmat2[i]', α, false)
    end
    dhZσ_λ = [dh.(v) for v ∈ Zσ_λ]
    Δ2generic!.(cone.Δ2_h_Zσ, Zσ_λ, cone.hZσ_λ, dhZσ_λ)
    fill!(cone.σmat, 0)
    for i ∈ eachindex(blocks)
        Zmat2[i] .= cone.Δ2_h_Zσ[i] .* cone.DhZmeat[i]
        spectral_outer!(Zmat3[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        applykraus_adj!(cone.σmat2, cone.Zk[i], Hermitian(Zmat3[i]), cone.Zσmat[i])
        cone.σmat .+= cone.σmat2
    end
    smat_to_svec!(cone.dΨdσ, cone.σmat, cone.rt2)

    @. @views cone.grad[cone.ρ_idxs] = zi * cone.sα * cone.dΨdρ
    @. @views cone.grad[cone.σ_idxs] = zi * cone.sα * cone.dΨdσ

    ## logdet part of gradient
    ρ_λ, ρ_U = cone.ρ_fact
    cone.ρ_λ_inv .= inv.(ρ_λ)
    mul!(cone.ρmat, ρ_U, Diagonal(sqrt.(cone.ρ_λ_inv)))
    mul!(cone.ρ_inv, cone.ρmat, cone.ρmat')
    smat_to_svec!(cone.ρvec, cone.ρ_inv, cone.rt2)
    @views cone.grad[cone.ρ_idxs] .-= cone.ρvec

    σ_λ, σ_U = cone.σ_fact
    cone.σ_λ_inv .= inv.(σ_λ)
    mul!(cone.σmat, σ_U, Diagonal(sqrt.(cone.σ_λ_inv)))
    mul!(cone.σ_inv, cone.σmat, cone.σmat')
    smat_to_svec!(cone.σvec, cone.σ_inv, cone.rt2)
    @views cone.grad[cone.σ_idxs] .-= cone.σvec

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiRenyiQKDTri)
    @assert cone.grad_updated

    α = cone.α

    g(x) = x^α
    dg(x) = α * x^(α - 1)
    d2g(x) = α * (α - 1) * x^(α - 2)
    g̃(x) = α * x^α
    dg̃(x) = α^2 * x^(α - 1)
    h(x) = x^(1 / α - 1)
    dh(x) = (1 / α - 1) * x^(1 / α - 2)
    d2h(x) = (1 / α - 1) * (1 / α - 2) * x^(1 / α - 3)

    λ_ZGZ = cone.ZG_fact.S .^ 2
    Zσ_λ = [fact.values for fact ∈ cone.Zσ_fact]
    Δ2generic!(cone.Δ2_dg_ZGZ, λ_ZGZ, dg.(λ_ZGZ), d2g.(λ_ZGZ))
    Δ2generic!(cone.Δ2_g̃_ZGZ, λ_ZGZ, g̃.(λ_ZGZ), dg̃.(λ_ZGZ))
    Δ3generic!.(cone.Δ3_h_Zσ, cone.Δ2_h_Zσ, Zσ_λ, [d2h.(v) for v ∈ Zσ_λ])

    for i ∈ eachindex(cone.blocks)
        for j ∈ 1:cone.Zd[i]
            @views cone.Δ3_h_ZσW̃[i][:, :, j] .= cone.Δ3_h_Zσ[i][:, :, j] .* cone.DhZmeat[i]
        end
    end

    cone.hess_aux_updated = true
    return cone.hess_aux_updated
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiRenyiQKDTri)
    cone.hess_aux_updated || update_hess_aux(cone)

    dΨdρ = cone.dΨdρ
    dΨdσ = cone.dΨdσ
    sα = cone.sα
    (ρ_λ, ρ_U) = cone.ρ_fact
    (σ_λ, σ_U) = cone.σ_fact

    zi = inv(cone.z)

    # For each vector ξ do:
    @inbounds for i ∈ 1:size(arr, 2)
        @views ρ_arr = arr[cone.ρ_idxs, i]
        @views ρ_prod = prod[cone.ρ_idxs, i]
        @views σ_arr = arr[cone.σ_idxs, i]
        @views σ_prod = prod[cone.σ_idxs, i]

        # ∇hh * h_arr + ∇hρ * ρ_arr + ∇hσ * σ_arr
        prod[1, i] = abs2(zi) * (arr[1, i] - sα * dot(dΨdρ, ρ_arr) - sα * dot(dΨdσ, σ_arr))
        # h_arr/z^2 - sα/z^2 * ⟨∇ρ Ψ, ρ_arr⟩ - sα/z^2 * ⟨∇σ Ψ, σ_arr⟩

        # ∇ρh * h_arr + ∇ρρ * ρ_arr + ∇ρσ * σ_arr
        @. ρ_prod = -sα * prod[1, i] * dΨdρ
        # -sα/z^2 * h_arr * ∇ρ Ψ + 1/z^2 ⟨∇ρ Ψ, ρ_arr⟩ * ∇ρ Ψ + 1/z^2 ⟨∇σ Ψ, σ_arr⟩*∇ρ Ψ

        d2Ψdρ2!(cone.d2Ψdρ2vec, ρ_arr, cone)
        @. ρ_prod += sα * zi * cone.d2Ψdρ2vec # + sα/z ∇ρρ Ψ
        d2Ψdρσ!(cone.d2Ψdρ2vec, σ_arr, cone)
        @. ρ_prod += sα * zi * cone.d2Ψdρ2vec # + sα/z ∇ρσ Ψ

        # ∇σh * h_arr + ∇σρ * ρ_arr + ∇σσ * σ_arr
        @. σ_prod = -sα * prod[1, i] * dΨdσ
        # -sα/z^2 * h_arr * ∇σ Ψ + 1/z^2 ⟨∇ρ Ψ, ρ_arr⟩ * ∇σ Ψ + 1/z^2 ⟨∇σ Ψ, σ_arr⟩*∇ρ Ψ

        d2Ψdσρ!(cone.d2Ψdσ2vec, ρ_arr, cone)
        @. σ_prod += sα * zi * cone.d2Ψdσ2vec # + sα/z ∇σρ Ψ
        d2Ψdσ2!(cone.d2Ψdσ2vec, σ_arr, cone)
        @. σ_prod += sα * zi * cone.d2Ψdσ2vec # + sα/z ∇σσ Ψ

        # Hessian of logdet(ρ)
        svec_to_smat!(cone.ρmat, ρ_arr, cone.rt2)
        spectral_outer!(cone.ρmat3, ρ_U', Hermitian(cone.ρmat), cone.ρmat2)  # U' ξ U
        ldiv!(Diagonal(ρ_λ), cone.ρmat3)  # Λ^-1 U' ξ U
        rdiv!(cone.ρmat3, Diagonal(ρ_λ))  # Λ^-1 U' ξ U Λ^-1
        spectral_outer!(cone.ρmat3, ρ_U, Hermitian(cone.ρmat3), cone.ρmat2)  # U Λ^-1 U' ξ U Λ^-1 U'
        ρ_prod .+= smat_to_svec!(cone.ρvec, cone.ρmat3, cone.rt2)

        # Hessian of logdet(σ)
        svec_to_smat!(cone.σmat, σ_arr, cone.rt2)
        spectral_outer!(cone.σmat3, σ_U', Hermitian(cone.σmat), cone.σmat2)  # U' ξ U
        ldiv!(Diagonal(σ_λ), cone.σmat3)  # Λ^-1 U' ξ U
        rdiv!(cone.σmat3, Diagonal(σ_λ))  # Λ^-1 U' ξ U Λ^-1
        spectral_outer!(cone.σmat3, σ_U, Hermitian(cone.σmat3), cone.σmat2)  # U Λ^-1 U' ξ U Λ^-1 U'
        σ_prod .+= smat_to_svec!(cone.σvec, cone.σmat3, cone.rt2)
    end

    return prod
end

function d2Ψdρ2!(
    d2Ψdρ2vec::AbstractVector{T},
    ρ_arr::AbstractVector{T},
    cone::EpiRenyiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    ρ_arr_mat = svec_to_smat!(cone.ρmat, ρ_arr, cone.rt2)
    d2Ψdρ2 = cone.ρmat2
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Gk = cone.Gk

    #GG G' ∘ (Z_S^½ ⋅Z_S^½) ∘ Ddg(Z_S^½ Gρ Z_S^½)[⋅] ∘ (Z_S^½ ⋅Z_S^½) ∘ G
    U_ZGZ = cone.ZG_fact.U
    Gmat3 = U_ZGZ' * cone.sqrtShZσ
    if cone.is_G_identity
        spectral_outer!(Gmat2, Gmat3, Hermitian(ρ_arr_mat), Gmat4)
        Gmat .= cone.Δ2_dg_ZGZ .* Gmat2
        spectral_outer!(d2Ψdρ2, Gmat3', Hermitian(Gmat), Gmat4)
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), cone.Gρmat)
        spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
        Gmat .= cone.Δ2_dg_ZGZ .* Gmat2
        spectral_outer!(Gmat2, Gmat3', Hermitian(Gmat), Gmat4)
        applykraus_adj!(d2Ψdρ2, Gk, Hermitian(Gmat2), cone.Gρmat)
    end

    smat_to_svec!(d2Ψdρ2vec, d2Ψdρ2, cone.rt2)
    return d2Ψdρ2vec
end

function d2Ψdρσ!(
    d2Ψdρ2vec::AbstractVector{T},
    σ_arr::AbstractVector{T},
    cone::EpiRenyiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    σ_arr_mat = svec_to_smat!(cone.σmat, σ_arr, cone.rt2)
    d2Ψdρ2 = cone.ρmat2
    blocks = cone.blocks
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    ZGmat = cone.ZGmat
    ZGmat2 = cone.ZGmat2
    Zk = cone.Zk
    Gk = cone.Gk

    Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]
    U_ZGZ = cone.ZG_fact.U
    #GZ G' ∘ (Z_S^½ ⋅ Z_S^½) ∘ Dg̃(Z_S^½ Gρ Z_S^½)[Z_S^-½ S' ⋅S Z_S^-½] Dh(Zσ)[ ⋅] ∘ Z
    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', cone.invsqrtShZσ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zσ_U[i]', cone.S[blocks[i], :])
        end
        applykraus!(Zmat[i], Zk[i], Hermitian(σ_arr_mat), cone.Zσmat[i])
        spectral_outer!(Zmat2[i], Zσ_U[i]', Hermitian(Zmat[i]), Zmat3[i])
        Zmat[i] .= cone.Δ2_h_Zσ[i] .* Zmat2[i]
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end
    if !cone.is_S_identity
        spectral_outer!(Gmat3, cone.invsqrtShZσ, Hermitian(Gmat), Gmat4)
        spectral_outer!(Gmat2, U_ZGZ', Hermitian(Gmat3), Gmat4)
    else
        spectral_outer!(Gmat2, U_ZGZ', Hermitian(Gmat), Gmat4)
    end
    Gmat .= cone.Δ2_g̃_ZGZ .* Gmat2
    mul!(Gmat3, cone.sqrtShZσ, U_ZGZ)
    spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
    applykraus_adj!(d2Ψdρ2, Gk, Hermitian(Gmat2), cone.Gρmat)

    smat_to_svec!(d2Ψdρ2vec, d2Ψdρ2, cone.rt2)
    return d2Ψdρ2vec
end

function d2Ψdσρ!(
    d2Ψdσ2vec::AbstractVector{T},
    ρ_arr::AbstractVector{T},
    cone::EpiRenyiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    ρ_arr_mat = svec_to_smat!(cone.ρmat, ρ_arr, cone.rt2)
    d2Ψdσ2 = cone.σmat2
    blocks = cone.blocks
    S = cone.S
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    ZGmat = cone.ZGmat
    ZGmat2 = cone.ZGmat2
    Zk = cone.Zk
    Gk = cone.Gk

    #ZG Z' ∘ Dh(Zσ)[S Z_S^-½ ⋅Z_S^-½ S'] ∘ Dg̃(Z_S^½ Gρ Z_S^½)[Z_S^½ ⋅ Z_S^½] ∘ G
    Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]
    U_ZGZ = cone.ZG_fact.U
    Gmat3 = U_ZGZ' * cone.sqrtShZσ
    if cone.is_G_identity
        spectral_outer!(Gmat2, Gmat3, Hermitian(ρ_arr_mat), Gmat4)
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), cone.Gρmat)
        spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
    end
    Gmat .= cone.Δ2_g̃_ZGZ .* Gmat2
    spectral_outer!(Gmat2, U_ZGZ, Hermitian(Gmat), Gmat3)

    if !cone.is_S_identity
        spectral_outer!(Gmat, cone.invsqrtShZσ, Hermitian(Gmat2), Gmat4)
    end
    fill!(d2Ψdσ2, 0)
    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', cone.invsqrtShZσ[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat2), ZGmat2[i])
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        end
        Zmat2[i] .= cone.Δ2_h_Zσ[i] .* Zmat[i]
        spectral_outer!(Zmat[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat3[i])
        applykraus_adj!(cone.σmat3, Zk[i], Hermitian(Zmat[i]), cone.Zσmat[i])
        d2Ψdσ2 .+= cone.σmat3
    end
    smat_to_svec!(d2Ψdσ2vec, d2Ψdσ2, cone.rt2)

    return d2Ψdσ2vec
end

function d2Ψdσ2!(
    d2Ψdσ2vec::AbstractVector{T},
    σ_arr::AbstractVector{T},
    cone::EpiRenyiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    σ_arr_mat = svec_to_smat!(cone.σmat, σ_arr, cone.rt2)
    d2Ψdσ2 = cone.σmat2
    blocks = cone.blocks
    sqrtGρ = cone.sqrtGρ
    S = cone.S
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    ZGmat = cone.ZGmat
    ZGmat2 = cone.ZGmat2
    Zk = cone.Zk
    Gk = cone.Gk

    Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]
    #ZZ Z' ∘ Dh(Zσ)[S Gρ^½ ⋅ Gρ^½ S'] ∘ Ddg(Gρ^½ Z_S Gρ^½)[Gρ^½ S' ⋅S Gρ^½] ∘ Dh(Zσ)[⋅] ∘ Z
    fill!(Gmat, 0)
    U_GZG = cone.ZG_fact.V
    Δ2_dg_GZG = cone.Δ2_dg_ZGZ
    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(σ_arr_mat), cone.Zσmat[i])
        spectral_outer!(Zmat2[i], Zσ_U[i]', Hermitian(Zmat[i]), Zmat3[i])
        Zmat[i] .= cone.Δ2_h_Zσ[i] .* Zmat2[i]
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', sqrtGρ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])
        end
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end
    if cone.is_S_identity
        spectral_outer!(Gmat2, U_GZG', Hermitian(Gmat), Gmat4)
    else
        mul!(Gmat3, U_GZG', sqrtGρ)
        spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
    end
    Gmat .= Δ2_dg_GZG .* Gmat2
    if cone.is_S_identity
        spectral_outer!(Gmat2, U_GZG, Hermitian(Gmat), Gmat4)
    else
        spectral_outer!(Gmat2, Gmat3', Hermitian(Gmat), Gmat4)
    end
    fill!(d2Ψdσ2, 0)
    for i ∈ eachindex(blocks)
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat2), ZGmat2[i])
        Zmat2[i] .= cone.Δ2_h_Zσ[i] .* Zmat[i]
        spectral_outer!(Zmat[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat3[i])
        applykraus_adj!(cone.σmat3, Zk[i], Hermitian(Zmat[i]), cone.Zσmat[i])
        d2Ψdσ2 .+= cone.σmat3
    end

    #    + Z' ∘ D²h(Zσ)[ ⋅, S Gρ^½ dg(Gρ^½ Z_S Gρ^½) Gρ^½ S'] ∘ Z
    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(σ_arr_mat), cone.Zσmat[i])
        spectral_outer!(Zmat2[i], Zσ_U[i]', Hermitian(Zmat[i]), Zmat3[i])
        for j ∈ 1:cone.Zd[i]
            @views mul!(Zmat[i][:, j], cone.Δ3_h_ZσW̃[i][:, :, j], Zmat2[i][:, j])
        end
        Zmat2[i] .= Zmat[i]
        Zmat2[i] .+= Zmat[i]'
        spectral_outer!(Zmat[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat3[i])
        applykraus_adj!(cone.σmat3, Zk[i], Hermitian(Zmat[i]), cone.Zσmat[i])
        d2Ψdσ2 .+= cone.σmat3
    end

    smat_to_svec!(d2Ψdσ2vec, d2Ψdσ2, cone.rt2)
    return d2Ψdσ2vec
end

function update_hess(cone::EpiRenyiQKDTri)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    zi = inv(cone.z)
    α = cone.α
    Gρ = cone.Gρ
    Gk = cone.Gk
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    Zmat3 = cone.Zmat3
    sqrtGρ = cone.sqrtGρ
    blocks = cone.blocks
    S = cone.S
    big_ρmat = cone.big_ρmat
    big_σmat = cone.big_σmat
    ds_g̃_ZGZ = cone.ds_g̃_ZGZ
    ds_h_Zσ = cone.ds_h_Zσ
    sqrtShZσ = cone.sqrtShZσ

    g(x) = x^α
    dg(x) = α * x^(α - 1)
    d2g(x) = α * (α - 1) * x^(α - 2)
    g̃(x) = α * x^α
    dg̃(x) = α^2 * x^(α - 1)

    h(x) = x^(1 / α - 1)
    dh(x) = (1 / α - 1) * x^(1 / α - 2)
    d2h(x) = (1 / α - 1) * (1 / α - 2) * x^(1 / α - 3)

    H[1, 1] = abs2(zi) #∇hh = 1/z^2
    @views Hρρ = H[cone.ρ_idxs, cone.ρ_idxs]
    @views Hρσ = H[cone.ρ_idxs, cone.σ_idxs]
    @views Hσσ = H[cone.σ_idxs, cone.σ_idxs]

    @views @. H[1, cone.ρ_idxs] = -abs2(zi) * cone.sα * cone.dΨdρ #∇hρ = -sα/z² * ∇ρ Ψ
    @views @. H[1, cone.σ_idxs] = -abs2(zi) * cone.sα * cone.dΨdσ #∇hσ = -sα/z² * ∇σ Ψ
    @views mul!(Hρρ, cone.dΨdρ, cone.dΨdρ', abs2(zi), false) #∇ρρ = 1/z² * (∇ρ Ψ) * (∇ρ Ψ)'
    @views mul!(Hρσ, cone.dΨdρ, cone.dΨdσ', abs2(zi), false) #∇ρσ = 1/z² * (∇ρ Ψ) * (∇σ Ψ)'
    @views mul!(Hσσ, cone.dΨdσ, cone.dΨdσ', abs2(zi), false) #∇σσ = 1/z² * (∇σ Ψ) * (∇σ Ψ)'

    #GG G' ∘ (Z_S^½ ⋅Z_S^½) ∘ Ddg(Z_S^½ Gρ Z_S^½)[⋅] ∘ (Z_S^½ ⋅Z_S^½) ∘ G
    U_ZGZ = cone.ZG_fact.U
    for i ∈ eachindex(Gk)
        mul!(cone.Gρmat, sqrtShZσ, Gk[i])
        mul!(cone.Gρmatvec[i], U_ZGZ', cone.Gρmat)
    end
    d_spectral!(big_ρmat, cone.Δ2_dg_ZGZ, cone.Gρmatvec, Gmat2, Gmat3, cone.Gρmat, cone.ρmat, cone.rt2)
    @. Hρρ += zi * cone.sα * big_ρmat #∇ρρ += sα/z ∇ρρ Ψ

    #ZG Z' ∘ Dh(Zσ)[S Z_S^-½ ⋅Z_S^-½ S'] ∘ Dg̃(Z_S^½ Gρ Z_S^½)[Z_S^½ ⋅ Z_S^½) ∘ G
    Zσ_λ = [fact.values for fact ∈ cone.Zσ_fact]
    for i ∈ eachindex(cone.blocks)
        copyto!(cone.Zmat3[i], cone.Zσ_fact[i].vectors')
    end
    Zσ_Uadj = cone.Zmat3
    copyto!(Gmat, U_ZGZ')
    d_spectral!(ds_g̃_ZGZ, cone.Δ2_g̃_ZGZ, Gmat, Gmat2, Gmat3, cone.rt2)
    d_spectral!.(ds_h_Zσ, cone.Δ2_h_Zσ, Zσ_Uadj, Zmat, Zmat2, Ref(cone.rt2))
    fill!(cone.big_σGmat, 0)
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            mul!(cone.big_σZmat[i], cone.Zadj[i], ds_h_Zσ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], cone.invsqrtShZσ[blocks[i], :], cone.rt2)
            mul!(cone.big_σGmat, cone.big_σZmat[i], cone.big_ZGmat[i], true, true)
        end
    else
        symm_kron!(cone.big_Gmat, cone.invsqrtShZσ, cone.rt2)
        for i ∈ eachindex(blocks)
            mul!(cone.big_σZmat[i], cone.Zadj[i], ds_h_Zσ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], S[blocks[i], :], cone.rt2)
            mul!(cone.big_σGmat2, cone.big_σZmat[i], cone.big_ZGmat[i])
            mul!(cone.big_σGmat, cone.big_σGmat2, Hermitian(cone.big_Gmat), true, true)
        end
    end
    symm_kron!(cone.big_Gmat, cone.sqrtShZσ, cone.rt2)
    mul!(cone.big_σGmat2, cone.big_σGmat, ds_g̃_ZGZ)
    mul!(cone.big_σGmat, cone.big_σGmat2, Hermitian(cone.big_Gmat))
    mul!(cone.big_σρmat, cone.big_σGmat, cone.G)
    @. Hρσ += zi * cone.sα * cone.big_σρmat' #∇ρσ += sα/z ∇ρσ Ψ

    #ZZ Z' ∘ Dh(Zσ)[S Gρ^½ ⋅ Gρ^½ S'] ∘ Ddg(Gρ^½ Z_S Gρ^½)[Gρ^½ S' ⋅S Gρ^½] ∘ Dh(Zσ)[⋅] ∘ Z
    U_GZG = cone.ZG_fact.V
    Δ2_dg_GZG = cone.Δ2_dg_ZGZ

    fill!(cone.big_σGmat, 0)
    if cone.is_S_identity
        copyto!(Gmat, U_GZG')
        d_spectral!(cone.big_Gmat, Δ2_dg_GZG, Gmat, Gmat2, Gmat3, cone.rt2)
        for i ∈ eachindex(blocks)
            mul!(cone.big_σZmat[i], cone.Zadj[i], ds_h_Zσ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], sqrtGρ[blocks[i], :], cone.rt2)
            mul!(cone.big_σGmat, cone.big_σZmat[i], cone.big_ZGmat[i], true, true)
        end
        mul!(cone.big_σGmat2, cone.big_σGmat, cone.big_Gmat)
    else
        mul!(Gmat, U_GZG', sqrtGρ)
        d_spectral!(cone.big_Gmat, Δ2_dg_GZG, Gmat, Gmat2, Gmat3, cone.rt2)
        for i ∈ eachindex(blocks)
            mul!(cone.big_σZmat[i], cone.Zadj[i], ds_h_Zσ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], S[blocks[i], :], cone.rt2)
            mul!(cone.big_σGmat, cone.big_σZmat[i], cone.big_ZGmat[i], true, true)
        end
        mul!(cone.big_σGmat2, cone.big_σGmat, cone.big_Gmat)
    end
    mul!(big_σmat, cone.big_σGmat2, cone.big_σGmat')

    ##    + Z' ∘ D²h(Zσ)[ ⋅, S Gρ^½ dg(Gρ^½ Z_S Gρ^½) Gρ^½ S'] ∘ Z
    d2_spectral!.(cone.big_Zmat, Zσ_Uadj, cone.Δ3_h_ZσW̃, cone.Zmat, cone.Zmat2, Ref(cone.rt2))
    for i ∈ eachindex(blocks)
        mul!(cone.big_σZmat[i], cone.Zadj[i], cone.big_Zmat[i])
        mul!(big_σmat, cone.big_σZmat[i], cone.Z[i], true, true)
    end
    @. Hσσ += zi * cone.sα * big_σmat #∇σσ += sα/z ∇σσ Ψ

    #logdet part
    symm_kron!(cone.big_ρmat, cone.ρ_inv, cone.rt2) # ∇ρρ += skron(ρ⁻¹)
    Hρρ .+= cone.big_ρmat
    symm_kron!(cone.big_σmat, cone.σ_inv, cone.rt2) # ∇σσ += skron(σ⁻¹)
    Hσσ .+= cone.big_σmat
    cone.hess_updated = true
    return cone.hess
end

function update_dder3_aux(cone::EpiRenyiQKDTri)
    @assert !cone.dder3_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)

    α = cone.α
    d2g̃(x) = α^2 * (α - 1) * x^(α - 2)
    d3g(x) = α * (α - 1) * (α - 2) * x^(α - 3)

    λ_ZGZ = cone.ZG_fact.S .^ 2  # ZS^½ G ZS^½ = U Λ^2 U'
    Δ3generic!(cone.Δ3_g̃_ZGZ, cone.Δ2_g̃_ZGZ, λ_ZGZ, d2g̃.(λ_ZGZ))  # D^2 g̃
    Δ3generic!(cone.Δ3_dg_ZGZ, cone.Δ2_dg_ZGZ, λ_ZGZ, d3g.(λ_ZGZ))  # D^2 g'

    cone.dder3_aux_updated = true
    return
end

function d3Ψdρ3!(
    d3Ψdρ3vec::AbstractVector{T},
    ρ_arr_mat::AbstractMatrix{R},
    cone::EpiRenyiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    d3Ψdρ3 = cone.mat2

    blocks = cone.blocks
    sqrtGρ = cone.sqrtGρ
    invsqrtGρ = cone.invsqrtGρ
    sqrtShZσ = cone.sqrtShZσ
    invsqrtShZσ = cone.invsqrtShZσ
    S = cone.S
    Gmat = cone.Gmat
    Gmat2 = cone.Gmat2
    Gmat3 = cone.Gmat3
    Gmat4 = cone.Gmat4
    Gmat5 = cone.Gmat5
    DG = cone.Gmat6
    Zmat = cone.Zmat
    Zmat2 = cone.Zmat2
    DZ = cone.Zmat3
    UZξU = cone.Zmat4
    ZGmat = cone.ZGmat
    ZGmat2 = cone.ZGmat2
    Zk = cone.Zk
    Gk = cone.Gk

    Δ2_g̃_ZGZ = cone.Δ2_g̃_ZGZ
    Δ3_g̃_ZGZ = cone.Δ3_g̃_ZGZ
    Δ3_dg_ZGZ = cone.Δ3_dg_ZGZ

    Δ2_h_Zσ = cone.Δ2_h_Zσ
    Δ3_h_Zσ = cone.Δ3_h_Zσ

    Zσ_U = [fact.vectors for fact ∈ cone.Zσ_fact]

    U_ZGZ = cone.ZG_fact.U
    U_GZG = cone.ZG_fact.V

    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(ρ_arr_mat), cone.Zσmat[i])  # Z(ξ)
        spectral_outer!(UZξU[i], Zσ_U[i]', Hermitian(Zmat[i]), Zmat2[i])  # U_z' Z(ξ) U_z
    end

    # * GGG
    # ZS_Gξ = sqrtShZσ * Gξ * sqrtShZσ
    # DG .= sqrtShZσ * second_frechet(Δ3_dg_ZGZ, U_ZGZ, ZS_Gξ) * sqrtShZσ

    mul!(Gmat3, U_ZGZ', sqrtShZσ)
    if cone.is_G_identity
        spectral_outer!(Gmat5, Gmat3, Hermitian(ρ_arr_mat), Gmat2)
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), cone.Gρmat)  # G(ξ)
        spectral_outer!(Gmat5, Gmat3, Hermitian(Gmat), Gmat2)  # Gmat5 = U' ZS^½ G(ξ) ZS^½ U
    end

    # Second freched derivative
    @inbounds @views for k ∈ 1:cone.Gd
        for j ∈ 1:k
            # D^2 g'(ZS^½ G ZS^½)[ZS^½ G(ξ) ZS^½, ZS^½ G(ξ) ZS^½]
            Gmat[j, k] = 2 * dot(Gmat5[:, j], Diagonal(Δ3_dg_ZGZ[:, j, k]), Gmat5[:, k])
        end
    end
    # ZS^½ D^2 g'(ZS^½ G ZS^½)[ZS^½ G(ξ) ZS^½, ZS^½ G(ξ) ZS^½] ZS^½
    spectral_outer!(DG, Gmat3', Hermitian(Gmat), Gmat2)

    # * GZG term

    # Dmeat1 = invsqrtShZσ * S' * first_frechet(Δ2_h_Zσ, Zσ_U, Zξ) * S * invsqrtShZσ
    # DGZG = sqrtShZσ * second_frechet(Δ3_g̃_ZGZ, U_ZGZ, ZS_Gξ, Dmeat1) * sqrtShZσ

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        Zmat[i] .= Δ2_h_Zσ[i] .* UZξU[i]   # h[2] ⊙ U_z' Z(ξ) U_z
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', invsqrtShZσ[blocks[i], :])
        else
            # S has dim (n,k), G has dim (k,k), Z has dim (n,n)
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])  # ZGmat = U_z'S has dim (n,k)
        end
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end

    if cone.is_S_identity
        spectral_outer!(Gmat, U_ZGZ', Hermitian(Gmat), Gmat2)
    else
        mul!(Gmat3, U_ZGZ', invsqrtShZσ)
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)  # Gmat = U' ZS^-½ S' Dh(Z)[Z(ξ)] S ZS^-½ U
    end

    # Second freched derivative
    @inbounds @views for k ∈ 1:cone.Gd
        for j ∈ 1:k
            # Gmat5 = U' ZS^½ G(ξ) ZS^½ U can be reused
            Gmat2[j, k] = dot(Gmat[:, j], Diagonal(Δ3_g̃_ZGZ[:, j, k]), Gmat5[:, k])
            Gmat2[j, k] += dot(Gmat5[:, j], Diagonal(Δ3_g̃_ZGZ[:, j, k]), Gmat[:, k])
        end
    end
    # Gmat2 = D^2g̃(ZS^½ G ZS^½)[ZS^(-½) S' Dh(Z)[Z(ξ)] S ZS^(-½), ZS^½ G(ξ) ZS^½]
    mul!(Gmat3, sqrtShZσ, U_ZGZ)
    spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat2), Gmat)
    DG .+= 2 * Gmat2  # * GGZ + GZG terms

    # * ZGG term

    # second_der = second_frechet(Δ3_g̃_ZGZ, U_ZGZ, ZS_Gξ)
    # GGmeat = S * invrootZSρ * second_der * invrootZSρ * S'
    # DZGG = first_frechet(Δ2_h_Zσ, Zσ_U, GGmeat)

    # Second freched derivative
    @inbounds @views for k ∈ 1:cone.Gd
        for j ∈ 1:k
            # Gmat5 = U' ZS^½ G(ξ) ZS^½ U can be reused
            Gmat[j, k] = 2 * dot(Gmat5[:, j], Diagonal(Δ3_g̃_ZGZ[:, j, k]), Gmat5[:, k])
        end
    end

    if cone.is_S_identity
        spectral_outer!(Gmat, U_ZGZ, Hermitian(Gmat), Gmat2)
    else
        mul!(Gmat3, invsqrtShZσ, U_ZGZ)
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)
    end

    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', cone.invsqrtShZσ[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        end
        Zmat2[i] .= Δ2_h_Zσ[i] .* Zmat[i]
        spectral_outer!(DZ[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat[i])
    end

    # * GZZ (1st term)

    # Dmeat1 = rootGρ * S' * second_frechet(Δ3_h_Zσ, Zσ_U, Zξ) * S * rootGρ
    # DGZZ = invrootGρ * first_frechet(Δ2_g̃_GZG, U_GZG, Dmeat1) * invrootGρ

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                Zmat[i][j, k] = 2 * dot(UZξU[i][:, j], Diagonal(Δ3_h_Zσ[i][:, j, k]), UZξU[i][:, k])
            end
        end
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', sqrtGρ[blocks[i], :])  # U_z' G^½
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])  # U_z' S
        end
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end
    if cone.is_S_identity
        spectral_outer!(Gmat, U_GZG', Hermitian(Gmat), Gmat2)
    else
        mul!(Gmat3, U_GZG', sqrtGρ)
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)
    end
    Gmat2 .= Δ2_g̃_ZGZ .* Gmat  # Dg̃(G^(-½) ZS G^(-½))[G^(-½)S' D^2h[Z(ξ), Z(ξ)] S G^(-½)]

    mul!(Gmat3, cone.invsqrtGρ, U_GZG)
    spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat2), Gmat)  # G^(-½) Dg̃(G^½ ZS G^½)[G^½ S' D^2h[Z(ξ), Z(ξ)] S G^½] G^(-½)
    DG .+= Gmat2

    # * GZZ (2nd term)

    # Dmeat1 = rootGρ * S' * first_frechet(Δ2_h_Zσ, Zσ_U, Zξ) * S * rootGρ
    # DGZZ .+= invrootGρ * second_frechet(Δ3_g̃_GZG, U_GZG, Dmeat1) * invrootGρ

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        Zmat[i] .= Δ2_h_Zσ[i] .* UZξU[i]
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', sqrtGρ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])
        end
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end
    if cone.is_S_identity
        spectral_outer!(Gmat5, U_GZG', Hermitian(Gmat), Gmat2)
    else
        mul!(Gmat3, U_GZG', sqrtGρ)
        spectral_outer!(Gmat5, Gmat3, Hermitian(Gmat), Gmat2)
    end
    # Gmat5 = V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V

    @inbounds @views for k ∈ 1:cone.Gd
        for j ∈ 1:k
            Gmat[j, k] = 2 * dot(Gmat5[:, j], Diagonal(Δ3_g̃_ZGZ[:, j, k]), Gmat5[:, k])
        end
    end
    # g̃[2](Λ) ⊙ V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V
    mul!(Gmat3, invsqrtGρ, U_GZG)  # Gmat3 = G^(-½) V
    spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)  # G^(-½) V (g̃[2](Λ) ⊙ V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V) V' G^(-½)
    DG .+= Gmat

    # Sum DG to d3Ψdρ3
    if cone.is_G_identity
        d3Ψdρ3 .= DG
    else
        applykraus_adj!(d3Ψdρ3, Gk, Hermitian(DG), cone.Gρmat)
    end

    # * ZGZ (1st term)

    # invGHg = invrootGρ * Gξ * invrootGρ
    # Dmeat1 = S * rootGρ * first_frechet(Δ2_g̃_GZG, U_GZG, invGHg) * rootGρ * S'
    # DZGZ = second_frechet(Δ3_h_Zσ, Zσ_U, Zξ, Dmeat1)

    applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), cone.Gρmat)  # G(ξ)
    spectral_outer!(Gmat4, Gmat3', Hermitian(Gmat), Gmat2)  # Gmat4 = V' G^(-½) G(ξ) G^(-½) V

    Gmat .= Δ2_g̃_ZGZ .* Gmat4  # g̃[1]⊙(V' G^(-½) G(ξ) G^(-½) V)

    if cone.is_S_identity
        spectral_outer!(Gmat, U_GZG, Hermitian(Gmat), Gmat2)
    else
        mul!(Gmat3, sqrtGρ, U_GZG)  # Gmat3 = G^½ V
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)
    end

    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', sqrtGρ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])
        end
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])  # U_z' S G^½ Dg̃ G^½ S' U_z
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                # UZξU = U_z' Z(ξ) U_z
                Zmat2[i][j, k] = dot(UZξU[i][:, j], Diagonal(Δ3_h_Zσ[i][:, j, k]), Zmat[i][:, k])
                Zmat2[i][j, k] += dot(Zmat[i][:, j], Diagonal(Δ3_h_Zσ[i][:, j, k]), UZξU[i][:, k])
            end
        end
        spectral_outer!(Zmat2[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= 2 * Zmat2[i]
    end

    # * ZGZ (2nd term)

    # Dmeat1 = rootGρ * S' * first_frechet(Δ2_h_Zσ, Zσ_U, Zξ) * S * rootGρ
    # Dmeat2 = S * rootGρ * second_frechet(Δ3_g̃_GZG, U_GZG, Dmeat1, invGHg) * rootGρ * S'
    # DZGZ .+= first_frechet(Δ2_h_Zσ, Zσ_U, Dmeat2)

    @inbounds @views for k ∈ 1:cone.Gd
        for j ∈ 1:k
            # Gmat4 = V' G^(-½) G(ξ) G^(-½) V
            # Gmat5 = V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V
            Gmat[j, k] = dot(Gmat5[:, j], Diagonal(Δ3_g̃_ZGZ[:, j, k]), Gmat4[:, k])
            Gmat[j, k] += dot(Gmat4[:, j], Diagonal(Δ3_g̃_ZGZ[:, j, k]), Gmat5[:, k])
        end
    end
    if cone.is_S_identity
        spectral_outer!(Gmat, U_GZG, Hermitian(Gmat), Gmat2)
    else
        # Gmat3 = G^½ V
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)  # G^½ D^2g̃ G^½
    end

    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zσ_U[i]', sqrtGρ[blocks[i], :])  # ZGmat = U_z' * G^½
        else
            @views mul!(ZGmat[i], Zσ_U[i]', S[blocks[i], :])  # ZGmat = U_z' * S
        end
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])  # U_z' S G^½ Dg̃ G^½ S' U_z
        Zmat2[i] .= Δ2_h_Zσ[i] .* Zmat[i]
        spectral_outer!(Zmat2[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= 2 * Zmat2[i]
    end

    # * ZZZ 1st term

    # W = S * rootGρ * dg(GZG) * rootGρ * S'
    # DZZZ = third_frechet(Δ3_h_Zσ, Zσ_λ, d3h.(Zσ_λ), Zσ_U, W, Zξ)

    α = cone.α
    d3h(x) = (1 / α - 1) * (1 / α - 2) * (1 / α - 3) * x^(1 / α - 4)
    Zσ_λ = [fact.values for fact ∈ cone.Zσ_fact]

    for i ∈ eachindex(blocks)
        fill!(Zmat[i], 0)
        @inbounds for k ∈ 1:cone.Zd[i], j ∈ 1:k
            Δ4generic_ij!(cone.Δ4_ij_h_Zσ[i], j, k, Δ3_h_Zσ[i], Zσ_λ[i], d3h.(Zσ_λ[i]))
            for b ∈ 1:cone.Zd[i]
                for a ∈ 1:cone.Zd[i]
                    temp = 2 * cone.DhZmeat[i][j, b] * UZξU[i][b, a] * UZξU[i][a, k]
                    temp +=
                        2 *
                        UZξU[i][j, b] *
                        (cone.DhZmeat[i][b, a] * UZξU[i][a, k] + UZξU[i][b, a] * cone.DhZmeat[i][a, k])
                    Zmat[i][j, k] += cone.Δ4_ij_h_Zσ[i][b, a] * temp
                end
            end
        end
        spectral_outer!(Zmat[i], Zσ_U[i], Hermitian(Zmat[i]), Zmat2[i])
        DZ[i] .+= Zmat[i]
    end

    # * ZZZ 2nd - 3rd terms

    # Dmeat1 = sqrtGρ * S' * first_frechet(Δ2_h_Zσ, Zσ_U, Zξ) * S * sqrtGρ
    # Dmeat2 = S * sqrtGρ * first_frechet(Δ2_dg_GZG, U_GZG, Dmeat1) * sqrtGρ * S'
    # DZZZ .+= 2 * second_frechet(Δ3_h_Zσ, Zσ_U, Dmeat2, Zξ)

    # Gmat5 = V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V
    Gmat .= cone.Δ2_dg_ZGZ .* Gmat5

    if cone.is_S_identity
        spectral_outer!(Gmat, U_GZG, Hermitian(Gmat), Gmat2)
    else
        # Gmat3 = G^½ V
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)  # G^½ D^2g̃ G^½
    end

    for i ∈ eachindex(blocks)
        # ZGmat = U_z' * S
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])  # Zmat = U_z' S G^½ Dg' G^½ S' U_z
        # UZξU = U_z' Z(ξ) U_z
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                Zmat2[i][j, k] = dot(Zmat[i][:, j], Diagonal(Δ3_h_Zσ[i][:, j, k]), UZξU[i][:, k])
                Zmat2[i][j, k] += dot(UZξU[i][:, j], Diagonal(Δ3_h_Zσ[i][:, j, k]), Zmat[i][:, k])
            end
        end
        spectral_outer!(Zmat2[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= 2 * Zmat2[i]
    end

    # * ZZZ 5th term

    # Dmeat1 = rootGρ * S' * first_frechet(Δ2_h_Zσ, Zσ_U, Zξ) * S * rootGρ
    # Dmeat2 = S * rootGρ * second_frechet(Δ3_dg_GZG, U_GZG, Dmeat1) * rootGρ * S'
    # DZZZ .+= first_frechet(Δ2_h_Zσ, Zσ_U, Dmeat2)

    @inbounds @views for k ∈ 1:cone.Gd
        for j ∈ 1:k
            # Gmat5 = V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V
            Gmat[j, k] = 2 * dot(Gmat5[:, j], Diagonal(Δ3_dg_ZGZ[:, j, k]), Gmat5[:, k])
        end
    end
    if cone.is_S_identity
        spectral_outer!(Gmat, U_GZG, Hermitian(Gmat), Gmat2)
    else
        # Gmat3 = G^½ V
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)  # G^½ D^2g̃ G^½
    end
    for i ∈ eachindex(blocks)
        # ZGmat = U_z' * S
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        Zmat2[i] .= cone.Δ2_h_Zσ[i] .* Zmat[i]
        spectral_outer!(Zmat2[i], Zσ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= Zmat2[i]
    end

    # * ZZZ 4th term

    # Dmeat1 = rootGρ * S' * second_frechet(Δ3_h_Zσ, Zσ_U, Zξ) * S * rootGρ
    # Dmeat2 = S * rootGρ * first_frechet(Δ2_dg_GZG, U_GZG, Dmeat1) * rootGρ * S'
    # DZZZ .+= first_frechet(Δ2_h_Zσ, Zσ_U, Dmeat2)

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        # UZξU = U_z' Z(ξ) U_z
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                Zmat[i][j, k] = 2 * dot(UZξU[i][:, j], Diagonal(Δ3_h_Zσ[i][:, j, k]), UZξU[i][:, k])
            end
        end
        # Zmat = h[2] ⊙ U_z' Z(ξ) U_z
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end
    if cone.is_S_identity
        spectral_outer!(Gmat, U_GZG', Hermitian(Gmat), Gmat2)
    else
        # Gmat3 = G^½ V
        spectral_outer!(Gmat, Gmat3', Hermitian(Gmat), Gmat2) # V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V
    end

    Gmat2 .= cone.Δ2_dg_ZGZ .* Gmat  # Gmat5 = g'[1]⊙(V' G^½ S' U_z (h[2] ⊙ U_z' Z(ξ) U_z) U_z' S G^½ V')

    if cone.is_S_identity
        spectral_outer!(Gmat2, U_GZG, Hermitian(Gmat2), Gmat)
    else
        # Gmat3 = G^½ V
        spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat2), Gmat) # V' G^½ S' Dh(Z)[Z(ξ)] S G^½ V
    end

    for i ∈ eachindex(blocks)
        spectral_outer!(Zmat2[i], ZGmat[i], Hermitian(Gmat2), ZGmat2[i])
        Zmat[i] .= cone.Δ2_h_Zσ[i] .* Zmat2[i]
        spectral_outer!(Zmat[i], Zσ_U[i], Hermitian(Zmat[i]), Zmat2[i])
        DZ[i] .+= Zmat[i]
    end

    # * Apply kraus to DZ

    for i ∈ eachindex(blocks)
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(DZ[i]), cone.Zσmat[i])
        d3Ψdρ3 .+= cone.mat3
    end
    smat_to_svec!(d3Ψdρ3vec, d3Ψdρ3, cone.rt2)
    return d3Ψdρ3vec
end

function dder3(cone::EpiRenyiQKDTri{T,R}, dir::AbstractVector{T}) where {T<:Real,R<:RealOrComplex{T}}
    cone.dder3_aux_updated || update_dder3_aux(cone)

    dder3 = cone.dder3
    rt2 = cone.rt2
    zi = inv(cone.z)

    @views ρ_dir = dir[cone.ρ_idxs]
    ρ_dir_mat = cone.mat
    svec_to_smat!(ρ_dir_mat, ρ_dir, cone.rt2)

    d2Ψdρ2!(cone.d2Ψdρ2vec, ρ_dir_mat, cone) # ∇ρρ(u) * (:, ξ[ρ])

    const0 = zi * (dir[1] - cone.sα * dot(ρ_dir, cone.dΨdρ))  #  zi * ξ[1] - sα * zi * ∇ρΨ⋅ξ[ρ]
    const1 = zi * (abs2(const0) + zi * cone.sα * dot(ρ_dir, cone.d2Ψdρ2vec) / 2)  # zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2

    # h component of dder3
    dder3[1] = const1

    # ρ component of dder3
    @views dder3_ρ = dder3[cone.ρ_idxs]

    (ρ_λ, ρ_U) = cone.ρ_fact
    spectral_outer!(cone.mat2, ρ_U', Hermitian(ρ_dir_mat), cone.mat3)  # U' ξ U
    tempvec = cone.ρ_λ_inv
    tempvec .= sqrt.(ρ_λ)
    @. cone.mat2 /= tempvec' #  U' ξ U sqrt(Λ-1)
    ldiv!(Diagonal(ρ_λ), cone.mat2) # Λ-1 U' ξ U sqrt(Λ-1)
    mul!(cone.mat3, ρ_U, cone.mat2) # ρ-1 ξ U sqrt(Λ-1)
    mul!(cone.mat2, cone.mat3, cone.mat3')  # ρ-1 ξ ρ-1 ξ ρ-1
    smat_to_svec!(dder3_ρ, cone.mat2, rt2)

    @. dder3_ρ += cone.sα * zi * const0 * cone.d2Ψdρ2vec
    @. dder3_ρ -= cone.sα * const1 * cone.dΨdρ

    d3Ψdρ3vec = cone.d2Ψdρ2vec  # reusing variable to save memory
    d3Ψdρ3!(d3Ψdρ3vec, ρ_dir_mat, cone)
    @. dder3_ρ -= cone.sα * (zi / 2) * d3Ψdρ3vec

    return dder3  # -∇^3 barrier[ξ,ξ] / 2
end
