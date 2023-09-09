"""
    autodiff_jac(f, x)

Compute the Jacobian of `f` at `x` using automatic differentiation.
"""
function autodiff_jac(f, x)
    J = similar(x, length(f(x)), length(x))
    J .= ForwardDiff.jacobian(f, x)
    return J
end

"""
    finite_diff_jac(f, x)

Compute the Jacobian of `f` at `x` using forward finite differences.
"""
function finite_diff_jac(f, x)
    xᵢ₊₁ = similar(x)
    fᵢ = f(x)
    fᵢ₊₁ = similar(fᵢ)
    J = similar(x, length(fᵢ), length(x))

    h = 1e-8
    for j in eachindex(x)
        xᵢ₊₁ .= x
        xᵢ₊₁[j] += h
        fᵢ₊₁ .= f(xᵢ₊₁)
        @. J[:, j] = (fᵢ₊₁ - fᵢ) / h
    end

    return J
end

function jacobian_function(jacobian, autodiff, f, p)
    if jacobian !== nothing
        return x -> jacobian(x, p)
    else
        if autodiff
            return x -> autodiff_jac(f, x)
        else
            return x -> finite_diff_jac(f, x)
        end
    end
end
