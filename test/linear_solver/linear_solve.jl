using ContinuationSuite
using Test

@testset "Linear solve with LUSolver" begin
    A = [2 3; 1 -4]
    b = [5, -3]

    x = ContinuationSuite.linsolve(A, b, LUSolver())
    @test x ≈ [1, 1]
end
