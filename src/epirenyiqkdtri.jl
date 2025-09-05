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
    invsqrtGρ::Matrix{R}
    Zρ::Vector{Matrix{R}}
    hZρ::Vector{Matrix{R}}
    ShZρ::Matrix{R}
    sqrtShZρ::Matrix{R}
    invsqrtShZρ::Matrix{R}
    G::Matrix{T}
    S::Matrix{R}
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
    hZρ_λ::Vector{Vector{T}}
    z::T
    Δ2_dg_ZGZ::Matrix{T}
    Δ3_dg_ZGZ::Array{T,3}
    Δ2_g̃_ZGZ::Matrix{T}
    Δ3_g̃_ZGZ::Array{T,3}
    Δ2_h_Zρ::Vector{Matrix{T}}
    Δ3_h_Zρ::Vector{Array{T,3}}
    Δ3_h_ZρW̃::Vector{Array{R,3}}
    # Δ4_ij_h_Zρ::Vector{Matrix{T}}
    Δ4_h_Zρ::Vector{Array{T,4}}
    dΨdρ::Vector{T}
    d2Ψdρ2vec::Vector{T}
    d2Ψdρ2::Matrix{T}
    DhZmeat::Vector{Matrix{R}}
    ds_g̃_ZGZ::Matrix{T} #TODO check if it's being reused in dder3
    ds_h_Zρ::Vector{Matrix{T}}

    ZG::Matrix{R}
    #variables below are just scratch space
    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
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
    Zρmat::Vector{Matrix{R}}
    ZGmat::Vector{Matrix{R}}
    ZGmat2::Vector{Matrix{R}}

    vec::Vector{T}
    Gvec::Vector{T}
    Zvec::Vector{Vector{T}}

    big_ρmat::Matrix{T}
    big_Gmat::Matrix{T}
    big_Gρmat::Matrix{T}
    big_ρGmat::Matrix{T}
    big_ρGmat2::Matrix{T}
    big_Zmat::Vector{Matrix{T}}
    big_ZGmat::Vector{Matrix{T}}
    big_ρZmat::Vector{Matrix{T}}

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

        Gkraus = [R.(Gk) for Gk ∈ Gkraus]
        Zkraus = [R.(Zk) for Zk ∈ Zkraus]
        cone.Gk = Gkraus
        cone.Zkbig = Zkraus
        cone.Zk = [filter!(!iszero, [Zk[blocks[i], :] for Zk ∈ Zkraus]) for i ∈ 1:cone.nblocks]
        cone.is_G_identity = (cone.Gk == [I(cone.d)])
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

use_dder3(cone::EpiRenyiQKDTri) = true

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
    cone.invsqrtGρ = zeros(R, Gd, Gd)
    cone.Zρ = [zeros(R, s, s) for s ∈ Zd]
    cone.hZρ = [zeros(R, s, s) for s ∈ Zd]
    cone.ShZρ = zeros(R, Gd, Gd)
    cone.sqrtShZρ = zeros(R, Gd, Gd)
    cone.invsqrtShZρ = zeros(R, Gd, Gd)
    cone.ρ_inv = zeros(R, d, d)
    cone.dΨdρ = zeros(T, ρ_dim)
    cone.d2Ψdρ2vec = zeros(T, ρ_dim)
    cone.Δ2_dg_ZGZ = zeros(T, Gd, Gd)
    cone.Δ3_dg_ZGZ = zeros(T, Gd, Gd, Gd)
    cone.Δ2_g̃_ZGZ = zeros(T, Gd, Gd)
    cone.Δ3_g̃_ZGZ = zeros(T, Gd, Gd, Gd)
    cone.Δ2_h_Zρ = [zeros(T, s, s) for s ∈ Zd]
    cone.Δ3_h_Zρ = [zeros(T, s, s, s) for s ∈ Zd]
    cone.Δ3_h_ZρW̃ = [zeros(R, s, s, s) for s ∈ Zd]
    cone.Δ4_h_Zρ = [zeros(T, s, s, s, s) for s ∈ Zd]
    # cone.Δ4_ij_h_Zρ = [zeros(T, s, s) for s ∈ Zd]
    cone.d2Ψdρ2 = zeros(T, ρ_dim, ρ_dim)
    cone.ρ_λ_inv = zeros(T, d)
    cone.Gρ_λ_log = zeros(T, Gd)
    cone.Zρ_λ_log = [zeros(T, s) for s ∈ Zd]
    cone.hZρ_λ = [zeros(T, s) for s ∈ Zd]
    cone.DhZmeat = [zeros(R, s, s) for s ∈ Zd]

    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.Gmat = zeros(R, Gd, Gd)
    cone.Gmat2 = zeros(R, Gd, Gd)
    cone.Gmat3 = zeros(R, Gd, Gd)
    cone.Gmat4 = zeros(R, Gd, Gd)
    cone.Gmat5 = zeros(R, Gd, Gd)
    cone.Gmat6 = zeros(R, Gd, Gd)
    cone.Gρmat = zeros(R, Gd, d)
    cone.Gρmatvec = [zeros(R, Gd, d) for _ ∈ 1:length(cone.Gk)]
    cone.Zmat = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat2 = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat3 = [zeros(R, s, s) for s ∈ Zd]
    cone.Zmat4 = [zeros(R, s, s) for s ∈ Zd]
    cone.ZGmat = [zeros(R, s, Gd) for s ∈ Zd]
    cone.ZGmat2 = [zeros(R, s, Gd) for s ∈ Zd]
    cone.Zρmat = [zeros(R, s, d) for s ∈ Zd]
    cone.ZG = zeros(R, Gd, Gd)

    cone.vec = zeros(T, ρ_dim)
    cone.Gvec = zeros(T, Gρ_dim)
    cone.Zvec = [zeros(T, s) for s ∈ Zρ_dim]
    cone.ds_g̃_ZGZ = zeros(T, Gρ_dim, Gρ_dim)
    cone.ds_h_Zρ = [zeros(T, s, s) for s ∈ Zρ_dim]
    cone.big_ρmat = zeros(T, ρ_dim, ρ_dim)
    cone.big_Gmat = zeros(T, Gρ_dim, Gρ_dim)
    cone.big_Gρmat = zeros(T, Gρ_dim, ρ_dim)
    cone.big_ρGmat = zeros(T, ρ_dim, Gρ_dim)
    cone.big_ρGmat2 = zeros(T, ρ_dim, Gρ_dim)
    cone.big_Zmat = [zeros(T, s, s) for s ∈ Zρ_dim]
    cone.big_ZGmat = [zeros(T, s, Gρ_dim) for s ∈ Zρ_dim]
    cone.big_ρZmat = [zeros(T, ρ_dim, s) for s ∈ Zρ_dim]
    return
end

get_nu(cone::EpiRenyiQKDTri) = cone.d + 1

function set_initial_point!(arr::AbstractVector{T}, cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    d = cone.d
    blocks = cone.blocks
    ShZρ = cone.ShZρ
    sqrtShZρ = cone.sqrtShZρ
    invsqrtShZρ = cone.invsqrtShZρ
    hZρ = cone.hZρ
    hZρ_λ = cone.hZρ_λ
    Gmat = cone.Gmat
    Zmat = cone.Zmat

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
    mul!(Gmat, Gρ_U, Diagonal(fourthroot.(Gρ_λ)))
    mul!(cone.sqrtGρ, Gmat, Gmat')
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    for i ∈ eachindex(Zρ_λ)
        hZρ_λ[i] .= h.(Zρ_λ[i])
    end
    spectral_outer!.(hZρ, Zρ_U, hZρ_λ, Zmat)
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            @views ShZρ[blocks[i], blocks[i]] .= hZρ[i]
            mul!(Zmat[i], Zρ_U[i], Diagonal(fourthroot.(hZρ_λ[i])))
            @views mul!(sqrtShZρ[blocks[i], blocks[i]], Zmat[i], Zmat[i]')
        end
    else
        fill!(ShZρ, 0)
        for i ∈ eachindex(blocks)
            @views spectral_outer!(Gmat, cone.S[blocks[i], :]', Hermitian(hZρ[i]), cone.ZGmat[i])
            ShZρ .+= Gmat
        end
        ShZρ_λ, ShZρ_U = eigen(Hermitian(ShZρ))
        mul!(Gmat, ShZρ_U, Diagonal(fourthroot.(ShZρ_λ)))
        mul!(sqrtShZρ, Gmat, Gmat')
    end
    mul!(cone.ZG, cone.sqrtShZρ, cone.sqrtGρ)
    renyi = mapreduce(x -> x^(2 * cone.α), +, svdvals(cone.ZG))

    arr[1] = 0.5 * (cone.sα * renyi + sqrt(4 + renyi^2))
    return arr
end

function update_feas(cone::EpiRenyiQKDTri{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    @assert !cone.feas_updated
    @views ρ_vec = cone.point[cone.ρ_idxs]
    blocks = cone.blocks
    ShZρ = cone.ShZρ
    sqrtShZρ = cone.sqrtShZρ
    invsqrtShZρ = cone.invsqrtShZρ
    hZρ = cone.hZρ
    hZρ_λ = cone.hZρ_λ
    Gmat = cone.Gmat
    Zmat = cone.Zmat

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
            mul!(Gmat, Gρ_U, Diagonal(fourthroot.(Gρ_λ)))
            mul!(cone.sqrtGρ, Gmat, Gmat')
            mul!(Gmat, Gρ_U, Diagonal(map(inv ∘ fourthroot, Gρ_λ)))
            mul!(cone.invsqrtGρ, Gmat, Gmat')

            Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
            Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
            for i ∈ eachindex(Zρ_λ)
                hZρ_λ[i] .= h.(Zρ_λ[i])
            end
            spectral_outer!.(hZρ, Zρ_U, hZρ_λ, Zmat)
            if cone.is_S_identity
                for i ∈ eachindex(blocks)
                    @views ShZρ[blocks[i], blocks[i]] .= hZρ[i]
                    mul!(Zmat[i], Zρ_U[i], Diagonal(fourthroot.(hZρ_λ[i])))
                    @views mul!(sqrtShZρ[blocks[i], blocks[i]], Zmat[i], Zmat[i]')
                    mul!(Zmat[i], Zρ_U[i], Diagonal(map(inv ∘ fourthroot, hZρ_λ[i])))
                    @views mul!(invsqrtShZρ[blocks[i], blocks[i]], Zmat[i], Zmat[i]')
                end
            else
                fill!(ShZρ, 0)
                for i ∈ eachindex(blocks)
                    @views spectral_outer!(Gmat, cone.S[blocks[i], :]', Hermitian(hZρ[i]), cone.ZGmat[i])
                    ShZρ .+= Gmat
                end
                ShZρ_λ, ShZρ_U = eigen(Hermitian(ShZρ))
                mul!(Gmat, ShZρ_U, Diagonal(fourthroot.(ShZρ_λ)))
                mul!(sqrtShZρ, Gmat, Gmat')
                mul!(Gmat, ShZρ_U, Diagonal(map(inv ∘ fourthroot, ShZρ_λ)))
                mul!(invsqrtShZρ, Gmat, Gmat')
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
    mul!(Gmat, cone.sqrtShZρ, U_ZGZ)
    mul!(Gmat2, Gmat, Diagonal(cone.ZG_fact.S .^ (cone.α - 1)))
    mul!(Gmat3, Gmat2, Gmat2')
    if cone.is_G_identity
        cone.mat .= α .* Gmat3
    else
        applykraus_adj!(cone.mat, cone.Gk, Hermitian(Gmat3), cone.Gρmat)
        cone.mat .*= α
    end

    ## Z part of gradient
    U_GZG = cone.ZG_fact.V
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
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
        mul!(ZGmat2[i], Zρ_U[i]', ZGmat[i])
        mul!(cone.DhZmeat[i], ZGmat2[i], ZGmat2[i]', α, false)
    end
    dhZρ_λ = [dh.(v) for v ∈ Zρ_λ]
    Δ2generic!.(cone.Δ2_h_Zρ, Zρ_λ, cone.hZρ_λ, dhZρ_λ)
    for i ∈ eachindex(blocks)
        Zmat2[i] .= cone.Δ2_h_Zρ[i] .* cone.DhZmeat[i]
        spectral_outer!(Zmat3[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        applykraus_adj!(cone.mat2, cone.Zk[i], Hermitian(Zmat3[i]), cone.Zρmat[i])
        cone.mat .+= cone.mat2
    end

    smat_to_svec!(cone.dΨdρ, cone.mat, cone.rt2)

    @. @views cone.grad[cone.ρ_idxs] = zi * cone.sα * cone.dΨdρ

    ## logdet part of gradient
    ρ_λ, ρ_U = cone.ρ_fact
    cone.ρ_λ_inv .= inv.(ρ_λ)
    mul!(cone.mat, ρ_U, Diagonal(sqrt.(cone.ρ_λ_inv)))
    mul!(cone.ρ_inv, cone.mat, cone.mat')
    smat_to_svec!(cone.vec, cone.ρ_inv, cone.rt2)
    @views cone.grad[cone.ρ_idxs] .-= cone.vec

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
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Δ2generic!(cone.Δ2_dg_ZGZ, λ_ZGZ, dg.(λ_ZGZ), d2g.(λ_ZGZ))
    Δ2generic!(cone.Δ2_g̃_ZGZ, λ_ZGZ, g̃.(λ_ZGZ), dg̃.(λ_ZGZ))
    Δ3generic!.(cone.Δ3_h_Zρ, cone.Δ2_h_Zρ, Zρ_λ, [d2h.(v) for v ∈ Zρ_λ])

    for i ∈ eachindex(cone.blocks)
        for j ∈ 1:cone.Zd[i]
            @views cone.Δ3_h_ZρW̃[i][:, :, j] .= cone.Δ3_h_Zρ[i][:, :, j] .* cone.DhZmeat[i]
        end
    end

    cone.hess_aux_updated = true
    return cone.hess_aux_updated
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiRenyiQKDTri)
    cone.hess_aux_updated || update_hess_aux(cone)

    ρ_idxs = cone.ρ_idxs
    dΨdρ = cone.dΨdρ
    ρ_arr_mat = cone.mat
    sα = cone.sα

    zi = inv(cone.z)

    # For each vector ξ do:
    @inbounds for i ∈ 1:size(arr, 2)
        # ∇hh * arr_h + ∇hρ * arr_ρ
        @views ρ_arr = arr[ρ_idxs, i]
        @views ρ_prod = prod[ρ_idxs, i]
        prod[1, i] = abs2(zi) * (arr[1, i] - sα * dot(dΨdρ, ρ_arr))  # arr_h/z^2 -sα/z^2 * ⟨∇ρ Ψ, arr_ρ⟩
        # ∇ρh * arr_h + ∇ρρ * arr_ρ
        @. ρ_prod = -sα * prod[1, i] * dΨdρ # -sα/z^2 * arr_h*∇ρ Ψ + 1/z^2 ⟨∇ρ Ψ, arr_ρ⟩*∇ρ Ψ

        # + sα/z ∇ρρ Ψ
        svec_to_smat!(ρ_arr_mat, ρ_arr, cone.rt2)
        d2Ψdρ2!(cone.d2Ψdρ2vec, ρ_arr_mat, cone)
        @. ρ_prod += sα * zi * cone.d2Ψdρ2vec

        # Hessian of log(det(ρ))
        spectral_outer!(cone.mat3, cone.ρ_inv, Hermitian(ρ_arr_mat), cone.mat2)  # ρ^-1 ξ ρ^-1
        ρ_prod .+= smat_to_svec!(cone.vec, cone.mat3, cone.rt2)
    end

    return prod
end

function d2Ψdρ2!(
    d2Ψdρ2vec::AbstractVector{T},
    ρ_arr_mat::AbstractMatrix{R},
    cone::EpiRenyiQKDTri{T,R}
) where {T<:Real,R<:RealOrComplex{T}}
    d2Ψdρ2 = cone.mat2
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

    α = cone.α
    g(x) = x^α
    dg(x) = α * x^(α - 1)
    d2g(x) = α * (α - 1) * x^(α - 2)
    g̃(x) = α * x^α
    dg̃(x) = α^2 * x^(α - 1)

    h(x) = x^(1 / α - 1)
    dh(x) = (1 / α - 1) * x^(1 / α - 2)
    d2h(x) = (1 / α - 1) * (1 / α - 2) * x^(1 / α - 3)

    #GG G' ∘ (Z_S^½ ⋅Z_S^½) ∘ Ddg(Z_S^½ Gρ Z_S^½)[⋅] ∘ (Z_S^½ ⋅Z_S^½) ∘ G
    U_ZGZ = cone.ZG_fact.U
    Gmat3 = U_ZGZ' * cone.sqrtShZρ
    if cone.is_G_identity
        spectral_outer!(Gmat2, Gmat3, Hermitian(ρ_arr_mat), Gmat4)
        Gmat .= cone.Δ2_dg_ZGZ .* Gmat2
        spectral_outer!(cone.mat2, Gmat3', Hermitian(Gmat), Gmat4)
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), cone.Gρmat)
        spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
        Gmat .= cone.Δ2_dg_ZGZ .* Gmat2
        spectral_outer!(Gmat2, Gmat3', Hermitian(Gmat), Gmat4)
        applykraus_adj!(d2Ψdρ2, Gk, Hermitian(Gmat2), cone.Gρmat)
    end

    #ZG Z' ∘ Dh(Zρ)[S Z_S^-½ ⋅Z_S^-½ S'] ∘ Dg̃(Z_S^½ Gρ Z_S^½)[Z_S^½ ⋅ Z_S^½] ∘ G
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]
    Gmat3 = U_ZGZ' * cone.sqrtShZρ
    if cone.is_G_identity
        spectral_outer!(Gmat2, Gmat3, Hermitian(ρ_arr_mat), Gmat4)
    else
        applykraus!(Gmat, Gk, Hermitian(ρ_arr_mat), cone.Gρmat)
        spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
    end
    Gmat .= cone.Δ2_g̃_ZGZ .* Gmat2
    spectral_outer!(Gmat2, U_ZGZ, Hermitian(Gmat), Gmat3)

    if !cone.is_S_identity
        spectral_outer!(Gmat, cone.invsqrtShZρ, Hermitian(Gmat2), Gmat4)
    end
    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zρ_U[i]', cone.invsqrtShZρ[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat2), ZGmat2[i])
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        end
        Zmat2[i] .= cone.Δ2_h_Zρ[i] .* Zmat[i]
        spectral_outer!(Zmat[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat3[i])
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat[i]), cone.Zρmat[i])
        d2Ψdρ2 .+= cone.mat3
    end

    #GZ G' ∘ (Z_S^½ ⋅ Z_S^½) ∘ Dg̃(Z_S^½ Gρ Z_S^½)[Z_S^-½ S' ⋅S Z_S^-½] Dh(Zρ)[ ⋅] ∘ Z
    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(ρ_arr_mat), cone.Zρmat[i])
        spectral_outer!(Zmat2[i], Zρ_U[i]', Hermitian(Zmat[i]), Zmat3[i])
        Zmat[i] .= cone.Δ2_h_Zρ[i] .* Zmat2[i]
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end
    if !cone.is_S_identity
        spectral_outer!(Gmat3, cone.invsqrtShZρ, Hermitian(Gmat), Gmat4)
        spectral_outer!(Gmat2, U_ZGZ', Hermitian(Gmat3), Gmat4)
    else
        spectral_outer!(Gmat2, U_ZGZ', Hermitian(Gmat), Gmat4)
    end
    Gmat .= cone.Δ2_g̃_ZGZ .* Gmat2
    mul!(Gmat3, cone.sqrtShZρ, U_ZGZ)
    spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat), Gmat4)
    applykraus_adj!(cone.mat3, Gk, Hermitian(Gmat2), cone.Gρmat)
    d2Ψdρ2 .+= cone.mat3

    #ZZ Z' ∘ Dh(Zρ)[S Gρ^½ ⋅ Gρ^½ S'] ∘ Ddg(Gρ^½ Z_S Gρ^½)[Gρ^½ S' ⋅S Gρ^½] ∘ Dh(Zρ)[⋅] ∘ Z
    fill!(Gmat, 0)
    U_GZG = cone.ZG_fact.V
    Δ2_dg_GZG = cone.Δ2_dg_ZGZ
    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(ρ_arr_mat), cone.Zρmat[i])
        spectral_outer!(Zmat2[i], Zρ_U[i]', Hermitian(Zmat[i]), Zmat3[i])
        Zmat[i] .= cone.Δ2_h_Zρ[i] .* Zmat2[i]
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zρ_U[i]', sqrtGρ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])
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
    for i ∈ eachindex(blocks)
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat2), ZGmat2[i])
        Zmat2[i] .= cone.Δ2_h_Zρ[i] .* Zmat[i]
        spectral_outer!(Zmat[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat3[i])
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat[i]), cone.Zρmat[i])
        d2Ψdρ2 .+= cone.mat3
    end

    #    + Z' ∘ D²h(Zρ)[ ⋅, S Gρ^½ dg(Gρ^½ Z_S Gρ^½) Gρ^½ S'] ∘ Z
    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(ρ_arr_mat), cone.Zρmat[i])
        spectral_outer!(Zmat2[i], Zρ_U[i]', Hermitian(Zmat[i]), Zmat3[i])
        for j ∈ 1:cone.Zd[i]
            @views mul!(Zmat[i][:, j], cone.Δ3_h_ZρW̃[i][:, :, j], Zmat2[i][:, j])
        end
        Zmat2[i] .= Zmat[i]
        Zmat2[i] .+= Zmat[i]'
        spectral_outer!(Zmat[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat3[i])
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(Zmat[i]), cone.Zρmat[i])
        d2Ψdρ2 .+= cone.mat3
    end

    smat_to_svec!(d2Ψdρ2vec, d2Ψdρ2, cone.rt2)

    return d2Ψdρ2vec
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
    d2Ψdρ2 = cone.d2Ψdρ2
    ds_g̃_ZGZ = cone.ds_g̃_ZGZ
    ds_h_Zρ = cone.ds_h_Zρ
    sqrtShZρ = cone.sqrtShZρ

    g(x) = x^α
    dg(x) = α * x^(α - 1)
    d2g(x) = α * (α - 1) * x^(α - 2)
    g̃(x) = α * x^α
    dg̃(x) = α^2 * x^(α - 1)

    h(x) = x^(1 / α - 1)
    dh(x) = (1 / α - 1) * x^(1 / α - 2)
    d2h(x) = (1 / α - 1) * (1 / α - 2) * x^(1 / α - 3)

    H[1, 1] = abs2(zi) #∇hh = 1/z^2
    @views @. H[1, cone.ρ_idxs] = -abs2(zi) * cone.sα * cone.dΨdρ #∇hρ = -sα/z² * ∇ρ Ψ
    @views Hρ = H[cone.ρ_idxs, cone.ρ_idxs]
    @views mul!(Hρ, cone.dΨdρ, cone.dΨdρ', abs2(zi), false) #∇ρρ = 1/z² * (∇ρ Ψ) * (∇ρ Ψ)'

    #GG G' ∘ (Z_S^½ ⋅Z_S^½) ∘ Ddg(Z_S^½ Gρ Z_S^½)[⋅] ∘ (Z_S^½ ⋅Z_S^½) ∘ G
    U_ZGZ = cone.ZG_fact.U
    for i ∈ eachindex(Gk)
        mul!(cone.Gρmat, sqrtShZρ, Gk[i])
        mul!(cone.Gρmatvec[i], U_ZGZ', cone.Gρmat)
    end
    d_spectral!(d2Ψdρ2, cone.Δ2_dg_ZGZ, cone.Gρmatvec, Gmat2, Gmat3, cone.Gρmat, cone.mat, cone.rt2)

    #ZG Z' ∘ Dh(Zρ)[S Z_S^-½ ⋅Z_S^-½ S'] ∘ Dg̃(Z_S^½ Gρ Z_S^½)[Z_S^½ ⋅ Z_S^½) ∘ G
    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    for i ∈ eachindex(cone.blocks)
        copyto!(cone.Zmat3[i], cone.Zρ_fact[i].vectors')
    end
    Zρ_Uadj = cone.Zmat3
    copyto!(Gmat, U_ZGZ')
    d_spectral!(ds_g̃_ZGZ, cone.Δ2_g̃_ZGZ, Gmat, Gmat2, Gmat3, cone.rt2)
    d_spectral!.(ds_h_Zρ, cone.Δ2_h_Zρ, Zρ_Uadj, Zmat, Zmat2, Ref(cone.rt2))
    fill!(cone.big_ρGmat, 0)
    if cone.is_S_identity
        for i ∈ eachindex(blocks)
            mul!(cone.big_ρZmat[i], cone.Zadj[i], ds_h_Zρ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], cone.invsqrtShZρ[blocks[i], :], cone.rt2)
            mul!(cone.big_ρGmat, cone.big_ρZmat[i], cone.big_ZGmat[i], true, true)
        end
    else
        symm_kron!(cone.big_Gmat, cone.invsqrtShZρ, cone.rt2)
        for i ∈ eachindex(blocks)
            mul!(cone.big_ρZmat[i], cone.Zadj[i], ds_h_Zρ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], S[blocks[i], :], cone.rt2)
            mul!(cone.big_ρGmat2, cone.big_ρZmat[i], cone.big_ZGmat[i])
            mul!(cone.big_ρGmat, cone.big_ρGmat2, Hermitian(cone.big_Gmat), true, true)
        end
    end
    symm_kron!(cone.big_Gmat, cone.sqrtShZρ, cone.rt2)
    mul!(cone.big_ρGmat2, cone.big_ρGmat, ds_g̃_ZGZ)
    mul!(cone.big_ρGmat, cone.big_ρGmat2, Hermitian(cone.big_Gmat))
    mul!(cone.big_ρmat, cone.big_ρGmat, cone.G)
    d2Ψdρ2 .+= cone.big_ρmat
    d2Ψdρ2 .+= cone.big_ρmat'

    #ZZ Z' ∘ Dh(Zρ)[S Gρ^½ ⋅ Gρ^½ S'] ∘ Ddg(Gρ^½ Z_S Gρ^½)[Gρ^½ S' ⋅S Gρ^½] ∘ Dh(Zρ)[⋅] ∘ Z
    U_GZG = cone.ZG_fact.V
    Δ2_dg_GZG = cone.Δ2_dg_ZGZ

    fill!(cone.big_ρGmat, 0)
    if cone.is_S_identity
        copyto!(Gmat, U_GZG')
        d_spectral!(cone.big_Gmat, Δ2_dg_GZG, Gmat, Gmat2, Gmat3, cone.rt2)
        for i ∈ eachindex(blocks)
            mul!(cone.big_ρZmat[i], cone.Zadj[i], ds_h_Zρ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], sqrtGρ[blocks[i], :], cone.rt2)
            mul!(cone.big_ρGmat, cone.big_ρZmat[i], cone.big_ZGmat[i], true, true)
        end
        mul!(cone.big_ρGmat2, cone.big_ρGmat, cone.big_Gmat)
    else
        mul!(Gmat, U_GZG', sqrtGρ)
        d_spectral!(cone.big_Gmat, Δ2_dg_GZG, Gmat, Gmat2, Gmat3, cone.rt2)
        for i ∈ eachindex(blocks)
            mul!(cone.big_ρZmat[i], cone.Zadj[i], ds_h_Zρ[i])
            @views symm_kron_full!(cone.big_ZGmat[i], S[blocks[i], :], cone.rt2)
            mul!(cone.big_ρGmat, cone.big_ρZmat[i], cone.big_ZGmat[i], true, true)
        end
        mul!(cone.big_ρGmat2, cone.big_ρGmat, cone.big_Gmat)
    end
    mul!(d2Ψdρ2, cone.big_ρGmat2, cone.big_ρGmat', true, true)

    ##    + Z' ∘ D²h(Zρ)[ ⋅, S Gρ^½ dg(Gρ^½ Z_S Gρ^½) Gρ^½ S'] ∘ Z
    #TODO: incorporate Z and Zadj in d2_spectral!
    d2_spectral!.(cone.big_Zmat, Zρ_Uadj, cone.Δ3_h_ZρW̃, cone.Zmat, cone.Zmat2, Ref(cone.rt2))
    for i ∈ eachindex(blocks)
        mul!(cone.big_ρZmat[i], cone.Zadj[i], cone.big_Zmat[i])
        mul!(d2Ψdρ2, cone.big_ρZmat[i], cone.Z[i], true, true)
    end

    @. Hρ += zi * cone.sα * d2Ψdρ2 #∇ρρ += sα/z ∇ρρ Ψ
    #logdet part
    symm_kron!(cone.big_ρmat, cone.ρ_inv, cone.rt2) # ∇ρρ += skron(ρ⁻¹)
    Hρ .+= cone.big_ρmat
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

    Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    d3h(x) = (1 / α - 1) * (1 / α - 2) * (1 / α - 3) * x^(1 / α - 4)
    Δ4generic!.(cone.Δ4_h_Zρ, cone.Δ3_h_Zρ, Zρ_λ, [d3h.(v) for v ∈ Zρ_λ])

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
    sqrtShZρ = cone.sqrtShZρ
    invsqrtShZρ = cone.invsqrtShZρ
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

    Δ2_h_Zρ = cone.Δ2_h_Zρ
    Δ3_h_Zρ = cone.Δ3_h_Zρ

    Zρ_U = [fact.vectors for fact ∈ cone.Zρ_fact]

    U_ZGZ = cone.ZG_fact.U
    U_GZG = cone.ZG_fact.V

    for i ∈ eachindex(blocks)
        applykraus!(Zmat[i], Zk[i], Hermitian(ρ_arr_mat), cone.Zρmat[i])  # Z(ξ)
        spectral_outer!(UZξU[i], Zρ_U[i]', Hermitian(Zmat[i]), Zmat2[i])  # U_z' Z(ξ) U_z
    end

    # * GGG
    # ZS_Gξ = sqrtShZρ * Gξ * sqrtShZρ
    # DG .= sqrtShZρ * second_frechet(Δ3_dg_ZGZ, U_ZGZ, ZS_Gξ) * sqrtShZρ

    mul!(Gmat3, U_ZGZ', sqrtShZρ)
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

    # Dmeat1 = invsqrtShZρ * S' * first_frechet(Δ2_h_Zρ, Zρ_U, Zξ) * S * invsqrtShZρ
    # DGZG = sqrtShZρ * second_frechet(Δ3_g̃_ZGZ, U_ZGZ, ZS_Gξ, Dmeat1) * sqrtShZρ

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        Zmat[i] .= Δ2_h_Zρ[i] .* UZξU[i]   # h[2] ⊙ U_z' Z(ξ) U_z
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zρ_U[i]', invsqrtShZρ[blocks[i], :])
        else
            # S has dim (n,k), G has dim (k,k), Z has dim (n,n)
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])  # ZGmat = U_z'S has dim (n,k)
        end
        spectral_outer!(Gmat2, ZGmat[i]', Hermitian(Zmat[i]), ZGmat2[i])
        Gmat .+= Gmat2
    end

    if cone.is_S_identity
        spectral_outer!(Gmat, U_ZGZ', Hermitian(Gmat), Gmat2)
    else
        mul!(Gmat3, U_ZGZ', invsqrtShZρ)
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
    mul!(Gmat3, sqrtShZρ, U_ZGZ)
    spectral_outer!(Gmat2, Gmat3, Hermitian(Gmat2), Gmat)
    DG .+= 2 * Gmat2  # * GGZ + GZG terms


    # * ZGG term

    # second_der = second_frechet(Δ3_g̃_ZGZ, U_ZGZ, ZS_Gξ)
    # GGmeat = S * invrootZSρ * second_der * invrootZSρ * S'
    # DZGG = first_frechet(Δ2_h_Zρ, Zρ_U, GGmeat)

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
        mul!(Gmat3, invsqrtShZρ, U_ZGZ)
        spectral_outer!(Gmat, Gmat3, Hermitian(Gmat), Gmat2)
    end

    for i ∈ eachindex(blocks)
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zρ_U[i]', cone.invsqrtShZρ[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])
            spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])
        end
        Zmat2[i] .= Δ2_h_Zρ[i] .* Zmat[i]
        spectral_outer!(DZ[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat[i])
    end

    # * GZZ (1st term)

    # Dmeat1 = rootGρ * S' * second_frechet(Δ3_h_Zρ, Zρ_U, Zξ) * S * rootGρ
    # DGZZ = invrootGρ * first_frechet(Δ2_g̃_GZG, U_GZG, Dmeat1) * invrootGρ

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                Zmat[i][j,k] = 2 * dot(UZξU[i][:, j], Diagonal(Δ3_h_Zρ[i][:, j, k]), UZξU[i][:, k])
            end
        end
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zρ_U[i]', sqrtGρ[blocks[i], :])  # U_z' G^½
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])  # U_z' S
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

    # Dmeat1 = rootGρ * S' * first_frechet(Δ2_h_Zρ, Zρ_U, Zξ) * S * rootGρ
    # DGZZ .+= invrootGρ * second_frechet(Δ3_g̃_GZG, U_GZG, Dmeat1) * invrootGρ

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        Zmat[i] .= Δ2_h_Zρ[i] .* UZξU[i]
        if cone.is_S_identity
            @views mul!(ZGmat[i], Zρ_U[i]', sqrtGρ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])
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
    # DZGZ = second_frechet(Δ3_h_Zρ, Zρ_U, Zξ, Dmeat1)

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
            @views mul!(ZGmat[i], Zρ_U[i]', sqrtGρ[blocks[i], :])
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])
        end
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])  # U_z' S G^½ Dg̃ G^½ S' U_z
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                # UZξU = U_z' Z(ξ) U_z
                Zmat2[i][j, k] = dot(UZξU[i][:, j], Diagonal(Δ3_h_Zρ[i][:, j, k]), Zmat[i][:, k])
                Zmat2[i][j, k] += dot(Zmat[i][:, j], Diagonal(Δ3_h_Zρ[i][:, j, k]), UZξU[i][:, k])
            end
        end
        spectral_outer!(Zmat2[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= 2 * Zmat2[i]
    end

    # * ZGZ (2nd term)

    # Dmeat1 = rootGρ * S' * first_frechet(Δ2_h_Zρ, Zρ_U, Zξ) * S * rootGρ
    # Dmeat2 = S * rootGρ * second_frechet(Δ3_g̃_GZG, U_GZG, Dmeat1, invGHg) * rootGρ * S'
    # DZGZ .+= first_frechet(Δ2_h_Zρ, Zρ_U, Dmeat2)

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
            @views mul!(ZGmat[i], Zρ_U[i]', sqrtGρ[blocks[i], :])  # ZGmat = U_z' * G^½
        else
            @views mul!(ZGmat[i], Zρ_U[i]', S[blocks[i], :])  # ZGmat = U_z' * S
        end
        spectral_outer!(Zmat[i], ZGmat[i], Hermitian(Gmat), ZGmat2[i])  # U_z' S G^½ Dg̃ G^½ S' U_z
        Zmat2[i] .= Δ2_h_Zρ[i] .* Zmat[i]
        spectral_outer!(Zmat2[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= 2 * Zmat2[i]
    end

    # * ZZZ 1st term 

    # W = S * rootGρ * dg(GZG) * rootGρ * S'
    # DZZZ = third_frechet(Δ3_h_Zρ, Zρ_λ, d3h.(Zρ_λ), Zρ_U, W, Zξ)

    # α = cone.α
    # d3h(x) = (1 / α - 1) * (1 / α - 2) * (1 / α - 3) * x^(1 / α - 4)
    # Zρ_λ = [fact.values for fact ∈ cone.Zρ_fact]
    
    for i ∈ eachindex(blocks)
        fill!(Zmat[i], 0)
        @inbounds for k in 1:cone.Zd[i], j in 1:k
            # Δ4generic_ij!(cone.Δ4_ij_h_Zρ[i], j, k, Δ3_h_Zρ[i], Zρ_λ[i], d3h.(Zρ_λ[i]))
            for b ∈ 1:cone.Zd[i]
                for a ∈ 1:cone.Zd[i]
                    temp = 2 * cone.DhZmeat[i][j, b] * UZξU[i][b, a] * UZξU[i][a, k]
                    temp += 2 * UZξU[i][j, b] * (cone.DhZmeat[i][b, a] * UZξU[i][a, k] + UZξU[i][b, a] * cone.DhZmeat[i][a, k])
                    # Zmat[i][j, k] += cone.Δ4_ij_h_Zρ[i][b, a] * temp
                    Zmat[i][j, k] += cone.Δ4_h_Zρ[i][j, b, a, k] * temp
                end
            end
        end
        spectral_outer!(Zmat[i], Zρ_U[i], Hermitian(Zmat[i]), Zmat2[i])
        DZ[i] .+= Zmat[i]
    end

    # * ZZZ 2nd - 3rd terms

    # Dmeat1 = sqrtGρ * S' * first_frechet(Δ2_h_Zρ, Zρ_U, Zξ) * S * sqrtGρ
    # Dmeat2 = S * sqrtGρ * first_frechet(Δ2_dg_GZG, U_GZG, Dmeat1) * sqrtGρ * S'
    # DZZZ .+= 2 * second_frechet(Δ3_h_Zρ, Zρ_U, Dmeat2, Zξ)
    
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
                Zmat2[i][j, k] = dot(Zmat[i][:, j], Diagonal(Δ3_h_Zρ[i][:, j, k]), UZξU[i][:, k])
                Zmat2[i][j, k] += dot(UZξU[i][:, j], Diagonal(Δ3_h_Zρ[i][:, j, k]), Zmat[i][:, k])
            end
        end
        spectral_outer!(Zmat2[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= 2 * Zmat2[i]
    end

    # * ZZZ 5th term

    # Dmeat1 = rootGρ * S' * first_frechet(Δ2_h_Zρ, Zρ_U, Zξ) * S * rootGρ
    # Dmeat2 = S * rootGρ * second_frechet(Δ3_dg_GZG, U_GZG, Dmeat1) * rootGρ * S'
    # DZZZ .+= first_frechet(Δ2_h_Zρ, Zρ_U, Dmeat2)

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
        Zmat2[i] .= cone.Δ2_h_Zρ[i] .* Zmat[i]
        spectral_outer!(Zmat2[i], Zρ_U[i], Hermitian(Zmat2[i]), Zmat[i])
        DZ[i] .+= Zmat2[i]
    end

    # * ZZZ 4th term

    # Dmeat1 = rootGρ * S' * second_frechet(Δ3_h_Zρ, Zρ_U, Zξ) * S * rootGρ
    # Dmeat2 = S * rootGρ * first_frechet(Δ2_dg_GZG, U_GZG, Dmeat1) * rootGρ * S'
    # DZZZ .+= first_frechet(Δ2_h_Zρ, Zρ_U, Dmeat2)

    fill!(Gmat, 0)
    for i ∈ eachindex(blocks)
        # UZξU = U_z' Z(ξ) U_z
        @inbounds @views for k ∈ 1:cone.Zd[i]
            for j ∈ 1:k
                Zmat[i][j, k] = 2 * dot(UZξU[i][:, j], Diagonal(Δ3_h_Zρ[i][:, j, k]), UZξU[i][:, k])
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
        Zmat[i] .= cone.Δ2_h_Zρ[i] .* Zmat2[i]
        spectral_outer!(Zmat[i], Zρ_U[i], Hermitian(Zmat[i]), Zmat2[i])
        DZ[i] .+= Zmat[i]
    end

    # * Apply kraus to DZ

    for i ∈ eachindex(blocks)
        applykraus_adj!(cone.mat3, Zk[i], Hermitian(DZ[i]), cone.Zρmat[i])
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
    const1 = zi * (abs2(const0) + zi * cone.sα * 0.5 * dot(ρ_dir, cone.d2Ψdρ2vec))  # zi^3 * (ξ[1]^2 + (∇ρz⋅ξ[ρ])^2 + 2 * ξ[1] * ∇ρz⋅ξ[ρ]) - zi^2 * ∇2ρρ(z)⋅ξ[ρ]/2

    # h component of dder3
    dder3[1] = const1

    # ρ component of dder3
    @views dder3_ρ = dder3[cone.ρ_idxs]

    (ρ_λ, ρ_U) = cone.ρ_fact
    spectral_outer!(cone.mat2, ρ_U', Hermitian(ρ_dir_mat), cone.mat3)  # U' ξ U
    cone.ρ_λ_inv .= sqrt.(ρ_λ)
    @. cone.mat2 /= cone.ρ_λ_inv' #  U' ξ U sqrt(Λ-1)
    ldiv!(Diagonal(ρ_λ), cone.mat2) # Λ-1 U' ξ U sqrt(Λ-1)
    mul!(cone.mat3, cone.mat2, cone.mat2')  # Λ-1 U' ξ U Λ-1 U' ξ U Λ-1
    spectral_outer!(cone.mat3, ρ_U, Hermitian(cone.mat3), cone.mat2)  # mat2 = U Λ-1 U' ξ U Λ-1 U' ξ U Λ-1 U'
    smat_to_svec!(dder3_ρ, cone.mat3, rt2)

    @. dder3_ρ += cone.sα * zi * const0 * cone.d2Ψdρ2vec
    @. dder3_ρ -= cone.sα * const1 * cone.dΨdρ

    d3Ψdρ3vec = cone.d2Ψdρ2vec  # reusing variable to save memory
    d3Ψdρ3!(d3Ψdρ3vec, ρ_dir_mat, cone)
    @. dder3_ρ -= 0.5 * cone.sα * zi * d3Ψdρ3vec

    return dder3  # - 0.5 * ∇^3 barrier[ξ,ξ]
end
