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
import ConicQKD: svec, smat, skron, derivative_spectral_function!

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
        for dout ∈ (2, 4)
            K = randn(R, dout, din)
            @test skron(K) * Mvec ≈ svec(K * M * K')
            skr = zeros(T, length(Mvec), length(Mvec))
            Γ = Matrix(Hermitian(randn(T, dout, dout)))
            temp1 = zeros(R, dout, dout)
            temp2 = zeros(R, dout, dout)
            temp3 = zeros(R, dout, din)
            temp4 = zeros(R, din, din)
            derivative_spectral_function!(skr, Γ, K, temp1, temp2, temp3, temp4, sqrt(T(2)))
            @test skr * Mvec ≈ svec(K' * (Γ .* (K * M * K')) * K)
            Kvec = [randn(R, dout, din) for _ in 1:2]
            derivative_spectral_function!(skr, Γ, Kvec, temp1, temp2, temp3, temp4, sqrt(T(2)))
            @test skr * Mvec ≈ sum(svec(Kj' * (Γ .* (Ki * M * Ki')) * Kj) for Ki in Kvec, Kj in Kvec)
        end
    end
end;
