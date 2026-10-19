using Test
using LinearAlgebra
import ConicQKD: svec, smat, skron, d_spectral!, d2_spectral, Δ2pow!, Δ3pow!, Δ4pow_ij!

@testset "array tests" begin
    real_types = [Float64, Float32, BigFloat]
    for T ∈ real_types, R ∈ (T, Complex{T})
        din = 3
        M = Hermitian(randn(R, din, din))
        N = Hermitian(randn(R, din, din))
        Mvec = svec(M)
        Nvec = svec(N)
        @test dot(Mvec, Nvec) ≈ dot(M, N)
        @test smat(Mvec) ≈ M
        for dout ∈ (2, 3, 4)
            K = randn(R, dout, din)
            @test skron(K) * Mvec ≈ svec(K * M * K')
            skr = zeros(T, length(Mvec), length(Mvec))
            Δ2 = Matrix(Hermitian(randn(T, dout, dout)))
            temp1 = zeros(R, dout, dout)
            temp2 = zeros(R, dout, dout)
            temp3 = zeros(R, dout, din)
            temp4 = zeros(R, din, din)
            d_spectral!(skr, Δ2, K, temp1, temp2, temp3, temp4, sqrt(T(2)))
            @test skr * Mvec ≈ svec(K' * (Δ2 .* (K * M * K')) * K)
            Kvec = [randn(R, dout, din) for _ ∈ 1:2]
            d_spectral!(skr, Δ2, Kvec, temp1, temp2, temp3, temp4, sqrt(T(2)))
            @test skr * Mvec ≈ sum(svec(Kj' * (Δ2 .* (Ki * M * Ki')) * Kj) for Ki ∈ Kvec, Kj ∈ Kvec)
            if dout == 3
                Δ3 = Δ3pow!(zeros(T, dout, dout, dout), Δ2, sort(rand(T, dout)), T(2), T(1))
                W = randn(R, dout, dout)
                skr3 = d2_spectral(Δ3, K, W)
                M̃ = K * M * K'
                W̃ = K * W * K'
                temp = zeros(R, dout, dout)
                for i ∈ 1:dout
                    L = M̃[:, i] * W̃[:, i]'
                    temp .+= Δ3[:, :, i] .* (L + L')
                end
                @test skr3 * Mvec ≈ svec(K' * temp * K)
            end
        end
    end
end

@testset "Δ2, Δ3, Δ4 accuracy" begin
    for α ∈ (-1e-3, 1e-3, 1 - 1e-3)
        λ = [1e-6, 2e-6, 2e-6, 1.0, 1+1e-5, 2.0, 100., 10000.]
        d = length(λ)
        bigλ = big.(λ)
        bigα = big(α)
        bigΔ2 = Δ2pow!(zeros(BigFloat, d, d), bigλ, bigα, one(bigα))
        Δ2 = Δ2pow!(zeros(d, d), λ, α, one(α))
        @test norm(Δ2 ./ bigΔ2 .- 1, Inf) < 1e-13
        bigΔ3 = Δ3pow!(zeros(BigFloat, d, d, d), bigΔ2, bigλ, bigα, one(bigα))
        Δ3 = Δ3pow!(zeros(d, d, d), Δ2, λ, α, one(α))
        @test norm(Δ3 ./ bigΔ3 .- 1, Inf) < 2e-11
        norminf = 0.0
        for i ∈ 1:d, j ∈ 1:d
            Δ4ijp = Δ4pow_ij!(zeros(d, d), i, j, Δ3, λ, α, one(α))
            bigΔ4ij = Δ4pow_ij!(zeros(BigFloat, d, d), i, j, bigΔ3, bigλ, bigα, one(bigα))
            norminf = max(norminf, norm(Δ4ijp ./ bigΔ4ij .- 1, Inf))
        end
        @test norminf < 7e-11
    end
end
