"""
    solve(prob, method, x₀)

Solve the nonlinear problem `prob` using the nonlinear equation solver `method`,
starting from the initial guess `x₀`. The nonlinear equation solver is used to
solve the nonlinear equation `f(x, p) = 0`, where `f` is the residual function
defined in `prob`.
"""
function solve(prob::AbstractNonlinearProblem, method::NonlinearEqSolver, x₀)
    @unpack f, jac = prob
    return nlsolve(f, jac, x₀, method)
end
