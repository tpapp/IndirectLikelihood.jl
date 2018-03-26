@testset "vectorizing parameters" begin
    A = Matrix(reshape(1:9, 3, :))
    @test vec_parameters(UpperTriangular(A)) == [1, 4, 5, 7, 8, 9]
    @test vec_parameters(LowerTriangular(A)) == [1, 2, 3, 5, 6, 9]
    @test vec_parameters(Symmetric(A)) == [1, 4, 5, 7, 8, 9] # may change
    @test vec_parameters(A) == Vector(1:9)
    @test vec_parameters(1.0) == [1.0]
    @test vec_parameters((1.0, [2, 3])) == [1.0, 2.0, 3.0]
end

@testset "fallback methods" begin
    output = @capture_err begin
        @test_throws MethodError MLE("something", "nonsensical")
    end
    @test contains(output, "You need to define `MLE` with this model type.")
end
