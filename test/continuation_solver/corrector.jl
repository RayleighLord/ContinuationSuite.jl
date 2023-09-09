using ContinuationSuite
using LinearAlgebra
using Test

@testset "Newton corrector with explicit jacobian" begin
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

    u = [0.25, 0.5, 1.8]
    t = [0.0, 0.0, -1.0]

    u = ContinuationSuite.correct(u, u, t, 0.0, prob)

    @test norm(f(u[1:2], u[end], 0.0)) ≤ 1e-6
end

@testset "Newton corrector with autodiff jacobian" begin
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

    u = [0.25, 0.5, 1.8]
    t = [0.0, 0.0, -1.0]

    u = ContinuationSuite.correct(u, u, t, 0.0, prob)

    @test norm(f(u[1:2], u[end], 0.0)) ≤ 1e-6
end

@testset "Newton corrector with finite_diff jacobian" begin
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

    u = [0.25, 0.5, 1.8]
    t = [0.0, 0.0, -1.0]

    u = ContinuationSuite.correct(u, u, t, 0.0, prob)

    @test norm(f(u[1:2], u[end], 0.0)) ≤ 1e-6
end
