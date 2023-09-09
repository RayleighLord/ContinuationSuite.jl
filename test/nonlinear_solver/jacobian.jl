using ContinuationSuite
using Test

@testset "Square jacobian with finite differences" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
    jac(x, p) = [2x[1] 2x[2]; 1 1]

    x0 = [1.0 1.0]
    J_exact = jac(x0, 0.0)
    J_approx = finite_diff_jac(x -> f(x, 0.0), x0)

    @test J_exact≈J_approx atol=1e-6
end

@testset "Square jacobian with automatic differentiation" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
    jac(x, p) = [2x[1] 2x[2]; 1 1]

    x0 = [1.0 1.0]
    J_exact = jac(x0, 0.0)
    J_auto_diff = autodiff_jac(x -> f(x, 0.0), x0)

    @test J_exact≈J_auto_diff atol=1e-15
end

@testset "Rectangular jacobian with finite differences" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1; x[1]^3 + 3x[2]]
    jac(x, p) = [2x[1] 2x[2]; 1 1; 3x[1]^2 3]

    x0 = [1.0 1.0]
    J_exact = jac(x0, 0.0)
    J_approx = finite_diff_jac(x -> f(x, 0.0), x0)

    @test J_exact≈J_approx atol=1e-6
end

@testset "Rectangular jacobian with automatic differentiation" begin
    f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1; x[1]^3 + 3x[2]]
    jac(x, p) = [2x[1] 2x[2]; 1 1; 3x[1]^2 3]

    x0 = [1.0 1.0]
    J_exact = jac(x0, 0.0)
    J_auto_diff = autodiff_jac(x -> f(x, 0.0), x0)

    @test J_exact≈J_auto_diff atol=1e-15
end
