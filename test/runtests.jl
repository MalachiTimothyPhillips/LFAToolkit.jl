using Test, Documenter, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

@testset "LFAToolkit" begin

    # ---------------------------------------------------------------------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------------------------------------------------------------------
    doctest(LFAToolkit; manual = false);

    # ---------------------------------------------------------------------------------------------------------------------
    # Full example
    # ---------------------------------------------------------------------------------------------------------------------

    # setup
    mesh = Mesh2D(1.0, 1.0);
    basis = TensorH1LagrangeBasis(4, 4, 2, 1);

    function massweakform(u::Array{Float64,1}, w::Array{Float64})
        v = u * w[1]
        return [v]
    end

    # mass operator
    inputs = [
        OperatorField(basis, EvaluationMode.interpolation),
        OperatorField(basis, EvaluationMode.quadratureweights),
    ];
    outputs = [OperatorField(basis, EvaluationMode.interpolation)];
    mass = Operator(massweakform, mesh, inputs, outputs);

    # stencil computation
    stencil = getstencil(mass);
    stencil = getstencil(mass);

    # diffusion setup
    function diffusionweakform(du::Array{Float64,1}, w::Array{Float64})
        dv = du * w[1]
        return [dv]
    end

    # Diffusion operator
    inputs = [
        OperatorField(basis, EvaluationMode.gradient),
        OperatorField(basis, EvaluationMode.quadratureweights),
    ];
    outputs = [OperatorField(basis, EvaluationMode.gradient)];
    diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

    # stencil computation
    stencil = getstencil(diffusion);
    stencil = getstencil(diffusion);

end # testset

# ---------------------------------------------------------------------------------------------------------------------
