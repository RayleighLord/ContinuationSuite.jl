using ContinuationSuite
using Test

@testset "Continuation Suite" begin
    @testset "Linear Solvers" begin
        include("linear_solver/linear_solve.jl")
    end

    @testset "Jacobian Computation" begin
        include("nonlinear_solver/jacobian.jl")
    end

    @testset "Newton Solver" begin
        include("nonlinear_solver/newton.jl")
    end

    @testset "Broyden Solver" begin
        include("nonlinear_solver/broyden.jl")
    end

    @testset "Nonlinear Problem Solver" begin
        include("nonlinear_solver/nonlinear_solve.jl")
    end

    @testset "Predictor Methods" begin
        include("continuation_solver/predictor.jl")
    end

    @testset "Corrector Methods" begin
        include("continuation_solver/corrector.jl")
    end

    @testset "Continuation Problem Solver" begin
        include("continuation_solver/continuation_problems.jl")
    end

    @testset "Continuation Problem Function with Parameters" begin
        include("continuation_solver/continuation_with_params.jl")
    end
end
