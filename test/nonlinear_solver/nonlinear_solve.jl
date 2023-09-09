using ContinuationSuite
using Test

@testset "Solve nonlinear problem with parameters and explicit jacobian" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
    jac(x, p) = [2x[1] 2x[2]; 1 1]

    p = 0.0
    prob = NonlinearProblem(f, p; jac = jac)

    x0 = [0.25, 0.5]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [1.0, 0.0]
end

@testset "Solve nonlinear problem without parameters and explicit jacobian" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
    jac(x, p) = [2x[1] 2x[2]; 1 1]

    prob = NonlinearProblem(f; jac = jac)

    x0 = [0.25, 0.5]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [1.0, 0.0]
end

@testset "Solve nonlinear problem with parameters and finite_diff jacobian" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]

    p = 0.0
    prob = NonlinearProblem(f, p; autodiff = false)

    x0 = [0.25, 0.5]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [1.0, 0.0]
end

@testset "Solve nonlinear problem without parameters and finite_diff jacobian" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]

    prob = NonlinearProblem(f; autodiff = false)

    x0 = [0.25, 0.5]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [1.0, 0.0]
end

@testset "Solve nonlinear problem with parameters and automatic diff jacobian" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]

    p = 0.0
    prob = NonlinearProblem(f, p)

    x0 = [0.25, 0.5]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [1.0, 0.0]
end

@testset "Solve nonlinear problem without parameters and automatic diff jacobian" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]

    prob = NonlinearProblem(f)

    x0 = [0.25, 0.5]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [0.0, 1.0]

    x0 = [0.5, 0.25]

    sol = solve(prob, Newton(), x0)

    @test sol ≈ [1.0, 0.0]
end
