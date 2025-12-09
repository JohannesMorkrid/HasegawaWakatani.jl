using HasegawaWakatani
using Test

@testset "HasegawaWakatani" begin
# Write your tests here.
    include("domain_tests.jl")
    include("display_tests.jl")
end

# Test MMS1, MSS2, MSS3, perform_step!, get_cache, unpack_cache#