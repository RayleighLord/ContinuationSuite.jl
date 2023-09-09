using ContinuationSuite
using Test

@testset "Solve nonlinear system of equations with Newton" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
    jac(x, p) = [2x[1] 2x[2]; 1 1]

    x0 = [0.25, 0.5]

    sol = ContinuationSuite.nlsolve(x -> f(x, 0.0), x -> jac(x, 0.0), x0, Newton())

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = ContinuationSuite.nlsolve(x -> f(x, 0.0), x -> jac(x, 0.0), x0, Newton())

    @test sol ≈ [1.0, 0.0]
end
