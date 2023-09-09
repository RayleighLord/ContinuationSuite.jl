var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ContinuationSuite","category":"page"},{"location":"#ContinuationSuite","page":"Home","title":"ContinuationSuite","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ContinuationSuite.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ContinuationSuite]","category":"page"},{"location":"#ContinuationSuite.Broyden","page":"Home","title":"ContinuationSuite.Broyden","text":"Broyden{T <: Real, LS <: LinearSolver} <: NonlinearEqSolver\n\nThe Broyden's method for solving nonlinear equations. The nonlinear problem is solved iteratively using Broyden's method, which is a quasi-Newton method.\n\nBroyden's method approximates the Jacobian matrix and updates it at every iteration. The update formula for the Jacobian is\n\nJ_n+1 = J_n + ((f_n+1 - f_n) - J_n Δx_n) Δx_n  (Δx_n Δx_n)\n\nwhere J_n is the Jacobian matrix at the nth iteration and Δx_n = x_{n+1} - x_n is the change in the solution.\n\nFields\n\nϵₓ::T = 1e-8: The tolerance for the norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||\nϵᵣ::T = 1e-8: The tolerance for the norm of the residual ||f(xₙ)||\nmaxiters::Int = 20: The maximum number of iterations.\nlsolver::LS = LUSolver(): The linear solver to use to solve the linear system.\ninfo::Vector{Float64} = [0.0, 0.0, 0.0]: The information about the nonlinear solver.\ninfo[1]: The norm of the residual ||f(xₙ)||\ninfo[2]: The norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||\ninfo[3]: The number of iterations.\n\nExamples\n\nusing NumericalContinuation\n\nf(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]\njac(x, p) = [2x[1] 2x[2]; 1 1]\n\np = 0.0\n\nprob = NonlinearProblem(f, p; jac = jac)\n\nx0 = [0.25, 0.5]\nsol = solve(prob, Broyden(), x0)\nsol = solve(prob, Broyden(ϵₓ = 1e-10, maxiters = 100), x0) # with different parameters\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.ContinuationParameters","page":"Home","title":"ContinuationSuite.ContinuationParameters","text":"ContinuationParameters{T <: Real, P <: Predictor, NLS <: NonlinearEqSolver} <: AbstractCache\n\nContinuation parameters for the continuation problem. The fields are\n\nFields\n\n- `λmin::T`: minimum value of the continuation parameter `λ`. (default: -1.0)\n- `λmax::T`: maximum value of the continuation parameter `λ`. (default: 1.0)\n- `Δs::T`: continuation step size. (default: 0.1)\n- `direction::Symbol`: direction of the continuation. Can be `:forward` or `:backward`. (default: `:forward`)\n- `maxsteps::Int`: maximum number of continuation steps. (default: 100)\n- `predictor::P`: predictor method. (default: `PseudoArcLength()`)\n- `corrector::NLS`: corrector method. (default: `Newton()`)\n- `verbose::Bool`: whether to print information about the continuation process or not. (default: `false`)\n- `ncols::Int`: number of columns to use when printing information about the continuation process. (default: 2)\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.ContinuationProblem","page":"Home","title":"ContinuationSuite.ContinuationProblem","text":"ContinuationProblem(f, ContPars [, p]; jac = nothing, autodiff = true)\n\nDefines a Continuation Problem\n\nTo define a continuation problem, you need to provide the function f and the continuation parameters ContPars. Optionally, you can provide the Jacobian jac and additional parameters in p. The autodiff keyword argument is used to determine whether to use automatic differentiation or not. The default is true. This setting is ignored if the jacobian is provided.\n\nA continuation problem is of the form:\n\nf(x λ p) = 0\n\nwhere p is a parameter vector, x is the solution vector and λ is the continuation parameter.\n\nFields\n\n- `f`: The function defining the continuation problem\n- `jac`: The Jacobian of the function defining the continuation problem\n- `p`: The parameter vector\n- `ContPars`: The continuation parameters\n- `autodiff`: Whether to use automatic differentiation or not\n- `f_org`: The original function provided by the user\n- `jac_org`: The original Jacobian provided by the user\n\nExample of usage\n\nusing NumericalContinuation\n\nfunction f(x, λ, p)\n    x₁, x₂ = x\n    f₁ = x₁^2 + x₂^2 - λ\n    f₂ = x₂^2 - 2x₁ + 1\n    return [f₁, f₂]\nend\n\nfunction jac(x, λ, p)\n    x₁, x₂ = x\n    fₓ = [2x₁ 2x₂; -2 2x₂]\n    fλ = [-1; 0]\n    return [fₓ fλ]\nend\n\ncont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,\n                                   direction = :backward, predictor = PseudoArcLength(),\n                                   corrector = Newton(), verbose = false)\n\nprob = ContinuationProblem(f, cont_pars; autodiff = false)\nx₀, λ₀ = [1.0, -1.0], 2.0\n\nsol = continuation(prob, x₀, λ₀)\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.ContinuationSolution","page":"Home","title":"ContinuationSuite.ContinuationSolution","text":"ContinuationSolution(u, prob::ContinuationProblem)\n\nConstructs a solution object for a continuation problem.\n\nThe solution is represented by a vector of the parameter values λ and a matrix of solutions x with the different components.\n\nFields\n\nx: The solution matrix.\nλ: The parameter values.\nprob: The continuation problem.\nncols: The number of columns of the solution matrix x printed in the terminal.\n\nExample of usage\n\nusing NumericalContinuation\n\nfunction f(x, λ, p)\n    x₁, x₂ = x\n    f₁ = x₁^2 + x₂^2 - λ\n    f₂ = x₂^2 - 2x₁ + 1\n    return [f₁, f₂]\nend\n\ncont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,\n                                   direction = :backward, predictor = PseudoArcLength(),\n                                   corrector = Newton(), verbose = false)\n\nprob = ContinuationProblem(f, cont_pars; autodiff = false)\nx₀, λ₀ = [1.0, -1.0], 2.0\n\nsol = continuation(prob, x₀, λ₀)\nx, λ = sol.x, sol.λ\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.LUSolver","page":"Home","title":"ContinuationSuite.LUSolver","text":"LUSolver <: LinearSolver\n\nThe default linear solver just calls the backslash \\ operator to solve the linear system\n\nA * x = b\nx = A \\ b\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.Newton","page":"Home","title":"ContinuationSuite.Newton","text":"Newton{T <: Real, LS <: LinearSolver} <: NonlinearEqSolver\n\nThe Newton Raphson method for solving nonlinear equations. The nonlinear problem is solved iteratively using\n\nJ(x_n) Δx_n = -f(x_n)\n\nwhere J(x_n) is the Jacobian matrix of f(x_n) and Δx_n is the update to the solution at the nth iteration. The update is given by\n\nx_n+1 = x_n + Δx_n\n\nFields\n\nϵₓ::T = 1e-8: The tolerance for the norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||\nϵᵣ::T = 1e-8: The tolerance for the norm of the residual ||f(xₙ)||\nmaxiters::Int = 20: The maximum number of iterations.\nlsolver::LS = LUSolver(): The linear solver to use to solve the linear system.\ninfo::Vector{Float64} = [0.0, 0.0, 0.0]: The information about the nonlinear solver.\ninfo[1]: The norm of the residual ||f(xₙ)||\ninfo[2]: The norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||\ninfo[3]: The number of iterations.\n\nExamples\n\nusing NumericalContinuation\n\nf(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]\njac(x, p) = [2x[1] 2x[2]; 1 1]\n\np = 0.0\n\nprob = NonlinearProblem(f, p; jac = jac)\n\nx0 = [0.25, 0.5]\nsol = solve(prob, Newton(), x0)\nsol = solve(prob, Newton(ϵₓ = 1e-10, maxiters = 100), x0) # with different parameters\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.NonlinearProblem","page":"Home","title":"ContinuationSuite.NonlinearProblem","text":"NonlinearProblem(f, [, p]; jac = nothing, autodiff = true)\n\nDefines a Nonlinear System of Equations Problem\n\nTo define a nonlinear problem, you need to provide the function f. Optionally, you can provide the Jacobian jac and additional parameters in p. The autodiff keyword argument is used to determine whether to use automatic differentiation or not. The default is true. This setting is ignored if the jacobian is provided.\n\nA nonlinear problem is of the form:\n\nf(x p) = 0\n\nwhere p is a parameter vector and x is the solution vector.\n\nFields\n\n- `f`: The function defining the nonlinear problem\n- `jac`: The Jacobian of the function defining the nonlinear problem\n- `p`: The parameter vector\n- `autodiff`: Whether to use automatic differentiation or not\n- `f_org`: The original function provided by the user\n- `jac_org`: The original Jacobian provided by the user\n\nExample of usage\n\nusing NumericalContinuation\n\nf(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]\njac(x, p) = [2x[1] 2x[2]; 1 1]\n\np = 0.0\n\nprob = NonlinearProblem(f)\nprob = NonlinearProblem(f, p)\nprob = NonlinearProblem(f, p; jac = jac)\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.PseudoArcLength","page":"Home","title":"ContinuationSuite.PseudoArcLength","text":"PseudoArcLength <: TangentPredictor\n\nUses pseudo-arclength continuation to make a prediction for the next solution.\n\nFrom a known value of the solution uₛ₀, the prediction is updated by computing the tangent vector t and the next solution is computed as:\n\nuₛ = uₛ₀ + t Δs\n\nwhere Δs is the continuation step size in the pseudo-arclength. The tangent vector t is obtained from:\n\nt = frac(xₛ λₛ)(xₛ λₛ)\n\n\nwhere xₛ, λₛ is the solution of the linear system:\n\nfₓ xₛ + fλ λₛ = 0\n\nwhich is equivalent to finding the nullspace of the extended jacobian [fₓ, fλ].\n\nThe tangent vector of the previous step is used to determine the direction of the tangent vector t. If the dot product of the previous tangent vector and the current tangent vector is negative, the tangent vector is flipped.\n\n\n\n\n\n","category":"type"},{"location":"#ContinuationSuite.autodiff_jac-Tuple{Any, Any}","page":"Home","title":"ContinuationSuite.autodiff_jac","text":"autodiff_jac(f, x)\n\nCompute the Jacobian of f at x using automatic differentiation.\n\n\n\n\n\n","category":"method"},{"location":"#ContinuationSuite.continuation-Tuple{ContinuationSuite.AbstractContinuationProblem, Any, Any}","page":"Home","title":"ContinuationSuite.continuation","text":"continuation(prob::ContinuationProblem, x₀, λ₀)\n\nSolve the continuation problem prob starting from the initial solution x₀ and the initial parameter value λ₀. Return a ContinuationSolution object containing the solution.\n\nExample of usage\n\nusing NumericalContinuation\n\nfunction f(x, λ, p)\n    x₁, x₂ = x\n    f₁ = x₁^2 + x₂^2 - λ\n    f₂ = x₂^2 - 2x₁ + 1\n    return [f₁, f₂]\nend\n\ncont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,\n                                   direction = :backward, predictor = PseudoArcLength(),\n                                   corrector = Newton(), verbose = false)\n\nprob = ContinuationProblem(f, cont_pars; autodiff = false)\nx₀, λ₀ = [1.0, -1.0], 2.0\n\nsol = continuation(prob, x₀, λ₀)\n\n\n\n\n\n","category":"method"},{"location":"#ContinuationSuite.finite_diff_jac-Tuple{Any, Any}","page":"Home","title":"ContinuationSuite.finite_diff_jac","text":"finite_diff_jac(f, x)\n\nCompute the Jacobian of f at x using forward finite differences.\n\n\n\n\n\n","category":"method"},{"location":"#ContinuationSuite.solve-Tuple{ContinuationSuite.AbstractNonlinearProblem, ContinuationSuite.NonlinearEqSolver, Any}","page":"Home","title":"ContinuationSuite.solve","text":"solve(prob, method, x₀)\n\nSolve the nonlinear problem prob using the nonlinear equation solver method, starting from the initial guess x₀. The nonlinear equation solver is used to solve the nonlinear equation f(x, p) = 0, where f is the residual function defined in prob.\n\n\n\n\n\n","category":"method"}]
}
