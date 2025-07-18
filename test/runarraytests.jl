#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
run barrier tests
=#

using Test
using Printf
using LinearAlgebra
import Hypatia.Cones
import ConicQKD: svec, smat, skron, d_spectral!, d2_spectral, Δ3generic

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
                Δ3 = Δ3generic(Δ2, randn(T, dout), randn(T, dout))
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
end;
