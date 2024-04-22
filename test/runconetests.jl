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
import Hypatia.Cones
import ConicQKD
include(joinpath(@__DIR__, "cone.jl"))

function cone_types(T::Type{<:Real})
    cones_T = [
        ConicQKD.EpiQKDTri{T,T}
        ConicQKD.EpiQKDTri{T,Complex{T}}
    ]

    return cones_T
end

@testset "cone tests" begin
    println("starting oracle tests")
    @testset "oracle tests" begin
        real_types = [
            Float64
            # Float32,
            # BigFloat,
        ]
        @testset "$cone" for T in real_types, cone in cone_types(T)
            println("$cone")
            test_time = @elapsed test_oracles(cone)
            @printf("%8.2e seconds\n", test_time)
        end
    end

    println("\nstarting barrier tests")
    @testset "barrier tests" begin
        real_types = [
            Float64
            # Float32,
            # BigFloat,
        ]
        @testset "$cone" for T in real_types, cone in cone_types(T)
            println("$cone")
            test_time = @elapsed test_barrier(cone)
            @printf("%8.2e seconds\n", test_time)
        end
    end

    println("\nstarting time/allocation measurements")
    @testset "allocation tests" begin
        real_types = [
            Float64
            # Float32,
            # BigFloat,
        ]
        @testset "$cone" for T in real_types, cone in cone_types(T)
            println("\n$cone")
            test_time = @elapsed show_time_alloc(cone)
            @printf("%8.2e seconds\n", test_time)
        end
        println()
    end
end;
