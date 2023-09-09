using ContinuationSuite
using LinearAlgebra
using Test

@testset "Continuation with explicit jacobian" begin
    function f(x, λ, p)
        x₁, x₂ = x
        f₁ = x₁^2 + x₂^2 - λ
        f₂ = x₂^2 - 2x₁ + 1
        return [f₁, f₂]
    end

    function jac(x, λ, p)
        x₁, x₂ = x
        fₓ = [2x₁ 2x₂; -2 2x₂]
        fλ = [-1; 0]
        return [fₓ fλ]
    end

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,
        direction = :backward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    prob = ContinuationProblem(f, cont_pars; jac = jac)
    x₀, λ₀ = [1.0, -1.0], 2.0

    sol = continuation(prob, x₀, λ₀)

    @test norm(f(sol.x[:, 1], sol.λ[1], 0.0)) ≤ 1e-6
    @test norm(f(sol.x[:, end], sol.λ[end], 0.0)) ≤ 1e-6
end

@testset "Continuation with automatic differentiation for the jacobian" begin
    function f(x, λ, p)
        x₁, x₂ = x
        f₁ = x₁^2 + x₂^2 - λ
        f₂ = x₂^2 - 2x₁ + 1
        return [f₁, f₂]
    end

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,
        direction = :backward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    prob = ContinuationProblem(f, cont_pars; autodiff = true)
    x₀, λ₀ = [1.0, -1.0], 2.0

    sol = continuation(prob, x₀, λ₀)

    @test norm(f(sol.x[:, 1], sol.λ[1], 0.0)) ≤ 1e-6
    @test norm(f(sol.x[:, end], sol.λ[end], 0.0)) ≤ 1e-6
end

@testset "Continuation with finite differences for the jacobian" begin
    function f(x, λ, p)
        x₁, x₂ = x
        f₁ = x₁^2 + x₂^2 - λ
        f₂ = x₂^2 - 2x₁ + 1
        return [f₁, f₂]
    end

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,
        direction = :backward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    prob = ContinuationProblem(f, cont_pars; autodiff = false)
    x₀, λ₀ = [1.0, -1.0], 2.0

    sol = continuation(prob, x₀, λ₀)

    @test norm(f(sol.x[:, 1], sol.λ[1], 0.0)) ≤ 1e-6
    @test norm(f(sol.x[:, end], sol.λ[end], 0.0)) ≤ 1e-6
end
