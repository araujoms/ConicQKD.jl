#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
run subset of tests
=#

using Test
using Printf
using Hypatia

test_files = ["cone"]

println()
@info("starting all tests")
println()
timings = Dict{String, Float64}()

@testset "all tests" begin
    all_test_time = @elapsed for t in test_files
        @info("starting $t tests")
        test_time = @elapsed include("run$(t)tests.jl")
        flush(stdout)
        flush(stderr)
        @info("finished $t tests in $(@sprintf("%8.2e seconds", test_time))")
        println()
        timings[t] = test_time
    end

    @info("finished all tests in $(@sprintf("%8.2e seconds", all_test_time))")
    println("\ntest suite timings (seconds):")
    display(timings)
    println()
end

println();
