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
    msg(fname) = "You need to define `IndirectLikelihood.$(fname)` with this model type."

    output = @capture_err begin
        @test_throws MethodError MLE("something", "nonsensical")
    end
    @test contains(output, msg("MLE"))

    output = @capture_err begin
        @test_throws MethodError common_random(RNG, "undefined")
    end
    @test contains(output, msg("common_random"))

    output = @capture_err begin
        @test_throws MethodError loglikelihood("something", "nonsensical", 2)
    end
    @test contains(output, msg("loglikelihood"))

    output = @capture_err begin
        @test_throws MethodError simulate_data(RNG, "something", "nonsensical", 2)
    end
    @test contains(output, msg("simulate_data"))
end
