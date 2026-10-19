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

_numerical_type(::Type{T}) where {T} = T

"""
    svec(M::AbstractMatrix)

Produces the scaled vectorized version of a Hermitian matrix `M`. The transformation preserves inner products, i.e., ⟨M,N⟩ = ⟨svec(M),svec(N)⟩.
"""
function svec(M::AbstractMatrix{T}) where {T}#the weird stuff here is to make it work with JuMP variables
    d = size(M, 1)
    numericalT = _numerical_type(T)
    vec_dim = Cones.svec_length(numericalT, d)
    v = Vector{real(T)}(undef, vec_dim)
    root2 = sqrt(real(numericalT(2)))
    if numericalT <: Real
        Cones.smat_to_svec!(v, 1 * M, root2)
    else
        Cones._smat_to_svec_complex!(v, M, root2)
    end
    return v
end
export svec

"""
    smat(v::AbstractVector)

Maps a vector `v` back into a Hermitian matrix M such that svec(M) = `v`.
"""
function smat(v::AbstractVector{T}) where {T} #the weird stuff here is to make it work with JuMP variables
    n = length(v)
    is_complex = (n != 1 && n == isqrt(n)^2)
    d = is_complex ? Cones.svec_side(Complex, n) : Cones.svec_side(Real, n)
    mtype = is_complex ? promote_type(eltype(v), Complex{Int}) : real(eltype(v))
    M = Matrix{mtype}(undef, d, d)
    numericalT = _numerical_type(T)
    root2 = sqrt(real(numericalT(2)))
    if !is_complex
        Cones.svec_to_smat!(M, 1 * v, root2)
        LinearAlgebra.copytri!(M, 'U')
        return Symmetric(M)
    else
        Cones._svec_to_smat_complex!(M, v, root2)
        LinearAlgebra.copytri!(M, 'U', true)
        return Hermitian(M)
    end
end
export smat

"""
Computes `skr` such that `skr*svec(x) = svec(mat*x*mat')` for real `mat` and Hermitian `x`
"""
function symm_kron_full!(skr::AbstractMatrix{T}, mat::AbstractVecOrMat{T}, rt2::T) where {T<:Real}
    dout, din = size.(Ref(mat), (1, 2))

    col_idx = 1
    @inbounds for l ∈ 1:din
        for k ∈ 1:(l-1)
            row_idx = 1
            for j ∈ 1:dout
                for i ∈ 1:(j-1)
                    skr[row_idx, col_idx] = mat[i, k] * mat[j, l] + mat[i, l] * mat[j, k]
                    row_idx += 1
                end
                skr[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
                row_idx += 1
            end
            col_idx += 1
        end

        row_idx = 1
        for j ∈ 1:dout
            for i ∈ 1:(j-1)
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
    @inbounds for l ∈ 1:din
        for k ∈ 1:(l-1)
            row_idx = 1
            for j ∈ 1:dout
                for i ∈ 1:(j-1)
                    a = mat[i, k] * conj(mat[j, l])
                    b = conj(mat[i, l]) * mat[j, k]
                    Cones.spectral_kron_element!(skr, row_idx, col_idx, conj(a), conj(b))
                    row_idx += 2
                end
                c = rt2 * mat[j, k] * conj(mat[j, l])
                skr[row_idx, col_idx] = real(c)
                skr[row_idx, col_idx+1] = -imag(c)
                row_idx += 1
            end
            col_idx += 2
        end

        row_idx = 1
        for j ∈ 1:dout
            for i ∈ 1:(j-1)
                c = rt2 * mat[i, l] * conj(mat[j, l])
                skr[row_idx, col_idx] = real(c)
                skr[row_idx+1, col_idx] = imag(c)
                row_idx += 2
            end
            skr[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
        end
        col_idx += 1
    end

    return skr
end

"""
Computes the matrix representation of the linear map
ξ ↦ ∑ᵢⱼ Kⱼ'*(Δ2 .* (Kᵢ*ξ*Kᵢ')*Kⱼ
acting on svec(ξ). It corresponds to the Fréchet derivative of a spectral function
with first divided differences matrix Δ2(Λ) on the point ∑ᵢKᵢ'*Λ*Kᵢ
"""
function d_spectral!(
    skr::AbstractMatrix{T},
    Δ2::Matrix{T},
    K::Vector{Matrix{R}},
    temp1::Matrix{R},
    temp2::Matrix{R},
    temp3::Matrix{R},
    temp4::Matrix{R},
    rt2::T
) where {T<:Real,R<:RealOrComplex{T}}
    rt2i = inv(rt2)
    scals = (R <: Complex{T} ? [rt2i, -rt2i * im] : [rt2i]) # real and imag parts
    col_idx = 0
    @inbounds for j ∈ 1:size(K[1], 2)
        for i ∈ 1:(j-1), scal ∈ scals
            for k ∈ 1:length(K)
                @views mul!(temp1, K[k][:, j], K[k][:, i]', scal, k != 1)
            end
            @. temp2 = Δ2 * (temp1 + temp1')
            applykraus_adj!(temp4, K, Hermitian(temp2), temp3)
            col_idx += 1
            @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
        end

        for k ∈ 1:length(K)
            @views mul!(temp1, K[k][:, j], K[k][:, j]', true, k != 1)
        end
        @. temp2 = Δ2 * temp1
        applykraus_adj!(temp4, K, Hermitian(temp2), temp3)
        col_idx += 1
        @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
    end

    return skr
end

"""
Computes the matrix representation of the linear map
ξ ↦ K'*(Δ2 .* (K*ξ*K')*K
acting on svec(ξ). It corresponds to the Fréchet derivative a spectral function
with first divided differences matrix Δ2(Λ) on the point K'*Λ*K
"""
function d_spectral!(
    skr::AbstractMatrix{T},
    Δ2::Matrix{T},
    K::Matrix{R},
    temp1::Matrix{R},
    temp2::Matrix{R},
    temp3::Matrix{R},
    temp4::Matrix{R},
    rt2::T
) where {T<:Real,R<:RealOrComplex{T}}
    rt2i = inv(rt2)
    scals = (R <: Complex{T} ? [rt2i, -rt2i * im] : [rt2i]) # real and imag parts
    col_idx = 0
    @inbounds for j ∈ 1:size(K, 2)
        @views K_j = K[:, j]
        for i ∈ 1:(j-1), scal ∈ scals
            @views K_i = K[:, i]
            mul!(temp1, K_j, K_i', scal, false)
            @. temp2 = Δ2 * (temp1 + temp1')
            mul!(temp3, Hermitian(temp2), K)
            mul!(temp4, K', temp3)
            col_idx += 1
            @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
        end

        mul!(temp1, K_j, K_j')
        @. temp2 = Δ2 * temp1
        mul!(temp3, Hermitian(temp2), K)
        mul!(temp4, K', temp3)
        col_idx += 1
        @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
    end

    return skr
end

#useful when K is square
function d_spectral!(
    skr::AbstractMatrix{T},
    Δ2::Matrix{T},
    K::Union{Matrix{R},Vector{Matrix{R}}},
    temp1::Matrix{R},
    temp2::Matrix{R},
    rt2::T
) where {T<:Real,R<:RealOrComplex{T}}
    return d_spectral!(skr, Δ2, K, temp1, temp2, temp1, temp2, rt2)
end

for (matrixtype, wrapper) ∈ ((:AbstractMatrix, :identity), (:Symmetric, :Symmetric), (:Hermitian, :Hermitian))
    @eval begin
        function applykraus(K::Vector{<:AbstractMatrix{T}}, M::$matrixtype{S}) where {T,S}
            dout, din = size(K[1])
            TS = Base.promote_op(*, T, S)
            temp = Matrix{TS}(undef, dout, din)
            result = Matrix{TS}(undef, dout, dout)
            return $wrapper(applykraus!(result, K, M, temp))
        end
    end
end

#temp must have the same dimensions as K[1]
function applykraus!(result, K, X, temp)
    spectral_outer!(result, K[1], X, temp)
    for i ∈ 2:length(K)
        mul!(temp, K[i], X)
        mul!(result, temp, K[i]', true, true)
    end
    return result
end

#temp must have the same dimensions as K[1]
function applykraus_adj!(result, K, X, temp)
    spectral_outer!(result, K[1]', X, temp)
    for i ∈ 2:length(K)
        mul!(temp, X, K[i])
        mul!(result, K[i]', temp, true, true)
    end
    return result
end

# (a^α - b^α)/(a - b) = α * m^(α - 1) * ((1+r)^α - (1-r)^α)/(2*α*r)
# where m = (a+b)/2 and r = (a-b)/(a+b)
# this is an accurate version of ((1+r)^α - (1-r)^α)/(2*α*r)
powdiff(r::T, α::T) where {T<:Real} = ifelse(iszero(r), one(r), (expm1(α*log1p(r)) - expm1(α*log1p(-r)))/(2*α*r))

function Δ2pow!(Δ2::Matrix{T}, λ::Vector{T}, α::T, c::T) where {T<:Real}
    d = length(λ)

    coeff = c * α
    @inbounds for j ∈ 1:d
        λⱼ = λ[j]
        for i ∈ 1:(j-1)
            λᵢ = λ[i]
            s = λⱼ + λᵢ
            δ = λⱼ - λᵢ
            Δ2[i, j] = if δ < 0.95 * s
                coeff * (s/2)^(α-1) * powdiff(δ/s, α)
            else
                c * (λⱼ^α - λᵢ^α) / δ
            end
        end
        Δ2[j, j] = coeff * λ[j] ^ (α - 1)
    end

    # make symmetric
    LinearAlgebra.copytri!(Δ2, 'U')
    return Δ2
end

function Δ3pow!(Δ3::Array{T,3}, Δ2::Matrix{T}, λ::Vector{T}, α::T, c::T) where {T<:Real}
    rteps = cbrt(eps(T))
    d = length(λ)

    coeff = c * α * (α - 1) / 2

    @inbounds for k ∈ 1:d, j ∈ 1:k, i ∈ 1:j
        λⱼ = λ[j]
        λₖ = λ[k]
        λⱼₖ = λⱼ - λₖ
        if abs(λⱼₖ) < rteps * (λⱼ + λₖ)
            λᵢ = λ[i]
            λᵢⱼ = λᵢ - λⱼ
            if abs(λᵢⱼ) < rteps * (λᵢ + λⱼ)
                m = (λᵢ + λⱼ + λₖ) / 3
                t = coeff * m ^ (α - 2)
            else
                t = (Δ2[i, j] - Δ2[j, k]) / (λᵢ - λₖ)
            end
        else
            t = (Δ2[i, j] - Δ2[i, k]) / λⱼₖ
        end

        Δ3[i, j, k] = Δ3[i, k, j] = Δ3[j, i, k] = Δ3[j, k, i] = Δ3[k, i, j] = Δ3[k, j, i] = t
    end

    return Δ3
end

function Δ4pow_ij!(Δ4_ij::Matrix{T}, i::Int, j::Int, Δ3::Array{T,3}, λ::Vector{T}, α::T, c::T) where {T<:Real}
    rteps = eps(T)^(2/9)
    d = length(λ)
    λᵢ = λ[i]
    λⱼ = λ[j]

    coeff = c * α * (α - 1) * (α - 2) / 6

    @inbounds for l ∈ 1:d, k ∈ 1:l
        λₖ = λ[k]
        λₗ = λ[l]
        λᵢⱼ = λᵢ - λⱼ
        λᵢₖ = λᵢ - λₖ
        λᵢₗ = λᵢ - λₗ
        Bᵢⱼ = (abs(λᵢⱼ) < rteps * (λᵢ + λⱼ))
        Bᵢₖ = (abs(λᵢₖ) < rteps * (λᵢ + λₖ))
        Bᵢₗ = (abs(λᵢₗ) < rteps * (λᵢ + λₗ))

        if Bᵢⱼ && Bᵢₖ && Bᵢₗ
            m = (λᵢ + λⱼ + λₖ + λₗ)/4
            t =  coeff * m ^ (α - 3)
        elseif Bᵢₖ && Bᵢₗ
            t = (Δ3[i, k, l] - Δ3[j, k, l]) / λᵢⱼ
        elseif Bᵢₗ
            t = (Δ3[i, j, l] - Δ3[j, k, l]) / λᵢₖ
        else
            t = (Δ3[i, j, k] - Δ3[j, k, l]) / λᵢₗ
        end

        Δ4_ij[k, l] = t
        Δ4_ij[l, k] = t
    end

    return Δ4_ij
end

function d_spectral(Δ2::Matrix{T}, K::Matrix{R}) where {T<:Real,R<:RealOrComplex{T}}
    dout, din = size(K)
    d = Cones.svec_length(R, din)
    skr = zeros(T, d, d)
    temp1 = zeros(R, dout, dout)
    temp2 = zeros(R, dout, dout)
    temp3 = zeros(R, dout, din)
    temp4 = zeros(R, din, din)
    d_spectral!(skr, Δ2, K, temp1, temp2, temp3, temp4, sqrt(T(2)))
    return skr
end

function d2_spectral(Δ3::Array{T,3}, U::Matrix{R}, W::Matrix{R}) where {T<:Real,R<:RealOrComplex{T}}
    d = size(U, 2)
    W̃ = U * W * U'
    Δ3W̃ = Array{R,3}(undef, d, d, d)
    for i ∈ 1:d
        @views Δ3W̃[:, :, i] .= Δ3[:, :, i] .* W̃
    end
    dim = Cones.svec_length(R, d)
    skr = zeros(T, dim, dim)
    temp2 = similar(U)
    temp3 = similar(U)
    d2_spectral!(skr, U, Δ3W̃, temp2, temp3, sqrt(T(2)))
    return skr
end

"""
Computes the matrix representation of the linear map
ξ ↦ V' * 2 herm(∑ᵢ (Δ3[:,:,i] .* (V*W*V')) * V*ξ*V'|i⟩⟨i|) * V
acting on svec(ξ). It corresponds to the second Fréchet derivative a spectral function
with second divided differences matrix Δ3(Λ) on the point V'*Λ*V.
The variable Δ3W̃ is defined as Δ3W̃[:,:,i] .= Δ3[:,:,i] .* (V*W*V')
"""
function d2_spectral!(
    skr::Matrix{T},
    V::Matrix{R},
    Δ3W̃::Array{R,3},
    temp1::Matrix{R},
    temp2::Matrix{R},
    rt2::T
) where {T<:Real,R<:RealOrComplex{T}}
    d = size(V, 2)
    rt2i = inv(rt2)
    scals = (R <: Complex{T} ? [rt2i, -rt2i * im] : [rt2i])

    col_idx = 1
    @inbounds for j ∈ 1:d
        @views V_j = V[:, j]
        for i ∈ 1:(j-1), scal ∈ scals
            @views V_i = V[:, i]
            mul!(temp2, V_j, V_i', scal, false)
            @. temp1 = temp2 + temp2'
            for k ∈ 1:d
                @views mul!(temp2[:, k], Δ3W̃[:, :, k], temp1[:, k])
            end
            @. temp1 = temp2 + temp2'
            spectral_outer!(temp1, V', Hermitian(temp1), temp2)
            @views smat_to_svec!(skr[:, col_idx], temp1, rt2)
            col_idx += 1
        end

        mul!(temp1, V_j, V_j')
        for k ∈ 1:d
            @views mul!(temp2[:, k], Δ3W̃[:, :, k], temp1[:, k])
        end
        @. temp1 = temp2 + temp2'
        spectral_outer!(temp1, V', Hermitian(temp1), temp2)
        @views smat_to_svec!(skr[:, col_idx], temp1, rt2)
        col_idx += 1
    end
    return skr
end

function spectral_outer!(
    mat::AbstractMatrix{R},
    vecs::Union{Matrix{R},SubArray{R}},
    symm::Hermitian{R},
    temp::Matrix{R}
) where {R<:RealOrComplex}
    mul!(temp, vecs, symm)
    mul!(mat, temp, vecs')
    return mat
end

function spectral_outer!(
    mat::AbstractMatrix{R},
    vecs::Adjoint{R,<:Union{Matrix{R},SubArray{R}}},
    symm::Hermitian{R},
    temp::Matrix{R}
) where {R<:RealOrComplex}
    mul!(temp, symm, vecs')
    mul!(mat, vecs, temp)
    return mat
end
