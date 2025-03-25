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

"""
Computes the matrix representation of the linear map
ξ ↦ ∑ᵢⱼ Kⱼ'*(Γ .* (Kᵢ*ξ*Kᵢ')*Kⱼ
acting on svec(ξ). It corresponds to the Hessian of a spectral function
with first divided differences matrix Γ.
"""
function hessian_spectral_function!(
    skr::AbstractMatrix{T},
    Γ::Matrix{T},
    K::Vector{Matrix{R}},
    temp1::Matrix{R},
    temp2::Matrix{R},
    temp3::Matrix{R},
    temp4::Matrix{R},
    rt2::T
) where {T<:Real,R<:RealOrComplex{T}}
    @assert issymmetric(Γ) # must be symmetric (wrapper is less efficient)
    rt2i = inv(rt2)
    scals = (R <: Complex{T} ? [rt2i, rt2i * im] : [rt2i]) # real and imag parts
    col_idx = 0
    @inbounds for j in 1:size(K[1], 2)
        for i in 1:(j-1), scal in scals
            for k = 1:length(K)
                @views mul!(temp1, K[k][:, j], K[k][:, i]', scal, k != 1)
            end
            @. temp2 = Γ * (temp1 + temp1')
            applykraus_adj!(temp4, K, Hermitian(temp2), temp3)
            col_idx += 1
            @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
        end

        for k = 1:length(K)
            @views mul!(temp1, K[k][:, j], K[k][:, j]', true, k != 1)
        end
        @. temp2 = Γ * temp1
        applykraus_adj!(temp4, K, Hermitian(temp2), temp3)
        col_idx += 1
        @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
    end

    return skr
end

"""
Computes the matrix representation of the linear map
ξ ↦ K'*(Γ .* (K*ξ*K')*K
acting on svec(ξ). It corresponds to the Hessian of a spectral function
with first divided differences matrix Γ.
"""
function hessian_spectral_function!(
    skr::AbstractMatrix{T},
    Γ::Matrix{T},
    K::Matrix{R},
    temp1::Matrix{R},
    temp2::Matrix{R},
    temp3::Matrix{R},
    temp4::Matrix{R},
    rt2::T
) where {T<:Real,R<:RealOrComplex{T}}
    @assert issymmetric(Γ) # must be symmetric (wrapper is less efficient)
    rt2i = inv(rt2)
    scals = (R <: Complex{T} ? [rt2i, rt2i * im] : [rt2i]) # real and imag parts
    col_idx = 0
    @inbounds for j in 1:size(K, 2)
        @views K_j = K[:, j]
        for i in 1:(j-1), scal in scals
            @views K_i = K[:, i]
            mul!(temp1, K_j, K_i', scal, false)
            @. temp2 = Γ * (temp1 + temp1')
            mul!(temp3, Hermitian(temp2), K)
            mul!(temp4, K', temp3)
            col_idx += 1
            @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
        end

        mul!(temp1, K_j, K_j')
        @. temp2 = Γ * temp1
        mul!(temp3, Hermitian(temp2), K)
        mul!(temp4, K', temp3)
        col_idx += 1
        @views smat_to_svec!(skr[:, col_idx], temp4, rt2)
    end

    return skr
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

function Δ2generic!(Δ2::Matrix{T}, λ::Vector{T}, fλ::Vector{T}, dfλ::Vector{T}) where {T <: Real}
    rteps = sqrt(eps(T))
    d = length(λ)

    @inbounds for j in 1:d
        for i in 1:(j - 1)
            λ_ij = λ[i] - λ[j]
            if abs(λ_ij) < rteps
                Δ2[i, j] = 0.5*(dfλ[i] + dfλ[j])
            else
                Δ2[i, j] = (fλ[i] - fλ[j]) / λ_ij
            end
        end
        Δ2[j, j] = dfλ[j]
    end

    # make symmetric
    LinearAlgebra.copytri!(Δ2, 'U')
    return Δ2
end

