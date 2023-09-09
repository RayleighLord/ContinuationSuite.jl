using ContinuationSuite
using LinearAlgebra
using Test

@testset "Continuation with explicit jacobian" begin
    function f(x, λ, p)
        x₁, x₂, x₃ = x
        q, r, s = p
        f₁ = -x₁ - x₂
        f₂ = -λ * x₃ + r * x₂ + s * x₃^2 - x₃^2 * x₂
        f₃ = -q * (x₁ + x₃)
        return [f₁, f₂, f₃]
    end

    function jac(x, λ, p)
        x₁, x₂, x₃ = x
        q, r, s = p
        fₓ = [-1 -1 0;
            0 (r-x₃^2) (-λ + 2s * x₃-2x₃ * x₂);
            -q 0 -q]
        fλ = [0; -x₃; 0]
        return [fₓ fλ]
    end

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.0, Δs = 0.25,
        direction = :forward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    p = 1.2, 0.8, 0.8
    prob = ContinuationProblem(f, cont_pars, p; jac = jac)
    x₀, λ₀ = [-1.4, -1.4, -1.4], 0.0
    sol = continuation(prob, x₀, λ₀)

    @test norm(f(sol.x[:, 1], sol.λ[1], p)) ≤ 1e-6
    @test norm(f(sol.x[:, end], sol.λ[end], p)) ≤ 1e-6
end

@testset "Continuation with automatic differentiation for the jacobian" begin
    function f(x, λ, p)
        x₁, x₂, x₃ = x
        q, r, s = p
        f₁ = -x₁ - x₂
        f₂ = -λ * x₃ + r * x₂ + s * x₃^2 - x₃^2 * x₂
        f₃ = -q * (x₁ + x₃)
        return [f₁, f₂, f₃]
    end

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.0, Δs = 0.25,
        direction = :forward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    p = 1.2, 0.8, 0.8
    prob = ContinuationProblem(f, cont_pars, p; autodiff = true)
    x₀, λ₀ = [-1.4, -1.4, -1.4], 0.0
    sol = continuation(prob, x₀, λ₀)

    @test norm(f(sol.x[:, 1], sol.λ[1], p)) ≤ 1e-6
    @test norm(f(sol.x[:, end], sol.λ[end], p)) ≤ 1e-6
end

@testset "Continuation with finite differences for the jacobian" begin
    function f(x, λ, p)
        x₁, x₂, x₃ = x
        q, r, s = p
        f₁ = -x₁ - x₂
        f₂ = -λ * x₃ + r * x₂ + s * x₃^2 - x₃^2 * x₂
        f₃ = -q * (x₁ + x₃)
        return [f₁, f₂, f₃]
    end

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.0, Δs = 0.25,
        direction = :forward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    p = 1.2, 0.8, 0.8
    prob = ContinuationProblem(f, cont_pars, p; autodiff = false)
    x₀, λ₀ = [-1.4, -1.4, -1.4], 0.0
    sol = continuation(prob, x₀, λ₀)

    @test norm(f(sol.x[:, 1], sol.λ[1], p)) ≤ 1e-6
    @test norm(f(sol.x[:, end], sol.λ[end], p)) ≤ 1e-6
end
