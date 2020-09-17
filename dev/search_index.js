var documenterSearchIndex = {"docs":
[{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"A. Brandt. Multi-level adaptive solutions to boundary-value problems. Math. Comp., 31(138):33-390, 1977.","category":"page"},{"location":"private/#Private-API","page":"Private API","title":"Private API","text":"","category":"section"},{"location":"private/","page":"Private API","title":"Private API","text":"This page documents the private API of the LFAToolkit.","category":"page"},{"location":"private/#Mesh","page":"Private API","title":"Mesh","text":"","category":"section"},{"location":"private/","page":"Private API","title":"Private API","text":"LFAToolkit.Mesh","category":"page"},{"location":"private/#LFAToolkit.Mesh","page":"Private API","title":"LFAToolkit.Mesh","text":"Rectangular mesh with independent scaling in each dimesion\n\n\n\n\n\n","category":"type"},{"location":"private/#Basis","page":"Private API","title":"Basis","text":"","category":"section"},{"location":"private/","page":"Private API","title":"Private API","text":"LFAToolkit.Basis\nLFAToolkit.gaussquadrature\nLFAToolkit.lobattoquadrature\nLFAToolkit.getnumbernodes\nLFAToolkit.getnumberquadraturepoints\nLFAToolkit.getinterpolation\nLFAToolkit.getgradient\nLFAToolkit.getquadratureweights","category":"page"},{"location":"private/#LFAToolkit.Basis","page":"Private API","title":"LFAToolkit.Basis","text":"Finite element basis for function spaces and test spaces\n\n\n\n\n\n","category":"type"},{"location":"private/#LFAToolkit.gaussquadrature","page":"Private API","title":"LFAToolkit.gaussquadrature","text":"gaussquadrature(q)\n\nConstruct a Gauss-Legendre quadrature\n\nArguments:\n\nq: number of Gauss-Legendre points\n\nReturns:\n\nGauss-Legendre quadrature points and weights\n\nExample:\n\n# generate Gauss-Legendre points and weights\nquadraturepoints, quadratureweights = LFAToolkit.gaussquadrature(5);\n\n# verify\ntruepoints = [\n    -sqrt(5 + 2*sqrt(10/7))/3,\n    -sqrt(5 - 2*sqrt(10/7))/3,\n    0.0,\n    sqrt(5 - 2*sqrt(10/7))/3,\n    sqrt(5 + 2*sqrt(10/7))/3\n];\ntrueweights = [\n    (322-13*sqrt(70))/900,\n    (322+13*sqrt(70))/900,\n    128/225,\n    (322+13*sqrt(70))/900,\n    (322-13*sqrt(70))/900\n];\n\ndiff = truepoints - quadraturepoints;\n@assert abs(max(diff...)) < 1e-15\n\ndiff = trueweights - quadratureweights;\n@assert abs(abs(max(diff...))) < 1e-15\n\n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#LFAToolkit.lobattoquadrature","page":"Private API","title":"LFAToolkit.lobattoquadrature","text":"lobattoquadrature(q, weights)\n\nConstruct a Gauss-Lobatto quadrature\n\nArguments:\n\nq:       number of Gauss-Lobatto points\nweights: boolean flag indicating if quadrature weights are desired\n\nReturns:\n\nGauss-Lobatto quadrature points or points and weights\n\nExample:\n\n# generate Gauss-Lobatto points\nquadraturepoints = LFAToolkit.lobattoquadrature(5, false);\n\n# verify\ntruepoints = [-1.0, -sqrt(3/7), 0.0, sqrt(3/7), 1.0];\n\ndiff = truepoints - quadraturepoints;\n@assert abs(max(diff...)) < 1e-15\n\n# generate Gauss-Lobatto points and weights\nquadraturepoints, quadratureweights = LFAToolkit.lobattoquadrature(5, true);\n\n# verify\ntrueweights = [1/10, 49/90, 32/45, 49/90, 1/10];\n\ndiff = trueweights - quadratureweights;\n@assert abs(abs(max(diff...))) < 1e-15\n\n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#LFAToolkit.getnumbernodes","page":"Private API","title":"LFAToolkit.getnumbernodes","text":"getnumbernodes(basis)\n\nGet the number of nodes for the basis\n\nArguments:\n\nbasis: basis to compute number of nodes\n\nReturns:\n\nInteger number of basis nodes\n\nExample:\n\n# get number of nodes for basis\nbasis = TensorH1LagrangeBasis(4, 3, 2);\nnumbernodes = LFAToolkit.getnumbernodes(basis);\n\n# verify\n@assert numbernodes == 4^2\n\n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#LFAToolkit.getnumberquadraturepoints","page":"Private API","title":"LFAToolkit.getnumberquadraturepoints","text":"getnumberquadraturepoints(basis)\n\nGet the number of quadrature points for the basis\n\nArguments:\n\nbasis: basis to compute number of quadrature points\n\nReturns:\n\nInteger number of basis quadrature points\n\nExample:\n\n# get number of quadrature points for basis\nbasis = TensorH1LagrangeBasis(4, 3, 2);\nquadraturepoints = LFAToolkit.getnumberquadraturepoints(basis);\n    \n# verify\n@assert quadraturepoints == 3^2\n    \n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#LFAToolkit.getinterpolation","page":"Private API","title":"LFAToolkit.getinterpolation","text":"getinterpolation(basis)\n\nGet full interpolation matrix for basis\n\nArguments:\n\nbasis: basis to compute interpolation matrix\n\nReturns:\n\nBasis interpolation matrix\n\nExample:\n\n# test for all supported dimensions\nfor dimension in 1:3\n    # get basis interpolation matrix\n    basis = TensorH1LagrangeBasis(4, 3, dimension);\n    interpolation = LFAToolkit.getinterpolation(basis);\n\n    # verify\n    for i in 1:3^dimension\n        total = sum(interpolation[i, :]);\n        @assert abs(total - 1.0) < 1e-15\n    end\nend\n\n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#LFAToolkit.getgradient","page":"Private API","title":"LFAToolkit.getgradient","text":"getgradient(basis)\n\nGet full gradient matrix for basis\n\nArguments:\n\nbasis: basis to compute gradient matrix\n\nReturns:\n\nBasis gradient matrix\n\nExample:\n\n# test for all supported dimensions\nfor dimension in 1:3\n    # get basis gradient matrix\n    basis = TensorH1LagrangeBasis(4, 3, dimension);\n    gradient = LFAToolkit.getgradient(basis);\n\n    # verify\n    for i in 1:dimension*3^dimension\n        total = sum(gradient[i, :]);\n        @assert abs(total) < 1e-14\n    end\nend\n\n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#LFAToolkit.getquadratureweights","page":"Private API","title":"LFAToolkit.getquadratureweights","text":"getquadratureweights(basis)\n\nGet full quadrature weights vector for basis\n\nReturns:\n\nBasis quadrature weights vector\n\nArguments:\n\nbasis: basis to compute quadrature weights\n\nExample:\n\n# test for all supported dimensions\nfor dimension in 1:3\n    # get basis quadrature weights\n    basis = TensorH1LagrangeBasis(4, 3, dimension);\n    quadratureweights = LFAToolkit.getquadratureweights(basis);\n\n    # verify\n    trueweights1d = [5/9, 8/9, 5/9];\n    trueweights = [];\n    if dimension == 1\n        trueweights = trueweights1d;\n    elseif dimension == 2\n        trueweights = kron(trueweights1d, trueweights1d);\n    elseif dimension == 3\n        trueweights = kron(trueweights1d, trueweights1d, trueweights1d);\n    end\n\n    diff = trueweights - quadratureweights;\n    @assert abs(abs(max(diff...))) < 1e-15\nend\n    \n# output\n\n\n\n\n\n\n","category":"function"},{"location":"private/#Operator","page":"Private API","title":"Operator","text":"","category":"section"},{"location":"private/","page":"Private API","title":"Private API","text":"LFAToolkit.getstencil","category":"page"},{"location":"private/#LFAToolkit.getstencil","page":"Private API","title":"LFAToolkit.getstencil","text":"getstencil(operator)\n\nCompute or retrieve the stencil of operator for computing the symbol\n\nArguments:\n\noperator: operator to compute element stencil\n\nReturns:\n\nAssembled element matrix\n\nMass matrix example:\n\n# setup\nmesh = Mesh2D(1.0, 1.0);\nbasis = TensorH1LagrangeBasis(4, 4, 2);\n    \nfunction massweakform(u::Array{Float64}, w::Array{Float64})\n    v = u * w[1]\n    return [v]\nend\n    \n# mass operator\ninputs = [\n    OperatorField(basis, [EvaluationMode.interpolation]),\n    OperatorField(basis, [EvaluationMode.quadratureweights]),\n];\noutputs = [OperatorField(basis, [EvaluationMode.interpolation])];\nmass = Operator(massweakform, mesh, inputs, outputs);\n    \n# stencil computation\nstencil = LFAToolkit.getstencil(mass);\n\n# verify\nu = ones(4*4);\nv = stencil * u;\n    \ntotal = sum(v);\n@assert abs(total - 4.0) < 1e-14\n\n# test caching\nstencil = LFAToolkit.getstencil(mass)\nv = stencil * u;\n    \ntotal = sum(v);\n@assert abs(total - 4.0) < 1e-14\n    \n# output\n    \n\nDiffusion matrix example:\n\n# setup\nmesh = Mesh2D(1.0, 1.0);\nbasis = TensorH1LagrangeBasis(4, 4, 2);\n    \nfunction diffusionweakform(du::Array{Float64}, w::Array{Float64})\n    dv = du * w[1]\n    return [dv]\nend\n    \n# diffusion operator\ninputs = [\n    OperatorField(basis, [EvaluationMode.gradient]),\n    OperatorField(basis, [EvaluationMode.quadratureweights]),\n];\noutputs = [OperatorField(basis, [EvaluationMode.gradient])];\ndiffusion = Operator(diffusionweakform, mesh, inputs, outputs);\n    \n# stencil computation\nstencil = LFAToolkit.getstencil(diffusion);\n    \n# verify\nu = ones(4*4);\nv = stencil * u;\n    \ntotal = sum(v);\n@assert abs(total) < 1e-14\n    \n# output\n    \n\n\n\n\n\n","category":"function"},{"location":"public/#Public-API","page":"Public API","title":"Public API","text":"","category":"section"},{"location":"public/","page":"Public API","title":"Public API","text":"This page documents the public API of the LFAToolkit.","category":"page"},{"location":"public/#Mesh","page":"Public API","title":"Mesh","text":"","category":"section"},{"location":"public/","page":"Public API","title":"Public API","text":"Mesh1D\nMesh2D\nMesh3D","category":"page"},{"location":"public/#LFAToolkit.Mesh1D","page":"Public API","title":"LFAToolkit.Mesh1D","text":"Mesh1D(dx)\n\nOne dimensional regular background mesh\n\nArguments:\n\ndx: deformation in x dimension\n\nReturns:\n\nOne dimensional mesh object\n\nExample:\n\nmesh = Mesh1D(1.0);\n\n@assert abs(mesh.dx - 1.0) < 1e-15\n\n# output\n\n\n\n\n\n\n","category":"type"},{"location":"public/#LFAToolkit.Mesh2D","page":"Public API","title":"LFAToolkit.Mesh2D","text":"Mesh2D(dx, dy)\n\nTwo dimensional regular background mesh\n\nArguments:\n\ndx: deformation in x dimension\ndy: deformation in y dimension\n\nReturns:\n\nTwo dimensional mesh object\n\nExample:\n\nmesh = Mesh2D(1.0, 0.5);\n\n@assert abs(mesh.dx - 1.0) < 1e-15\n@assert abs(mesh.dy - 0.5) < 1e-15\n\n# output\n\n\n\n\n\n\n","category":"type"},{"location":"public/#LFAToolkit.Mesh3D","page":"Public API","title":"LFAToolkit.Mesh3D","text":"Mesh3D(dx, dy, dz)\n\nThree dimensional regular background mesh\n\nArguments:\n\ndx: deformation in x dimension\ndy: deformation in y dimension\ndz: deformation in z dimension\n\nReturns:\n\nThree dimensional mesh object\n\nExample:\n\nmesh = Mesh3D(1.0, 0.5, 0.25);\n\n@assert abs(mesh.dx - 1.0) < 1e-15\n@assert abs(mesh.dy - 0.5) < 1e-15\n@assert abs(mesh.dz - 0.25) < 1e-15\n\n# output\n\n\n\n\n\n\n","category":"type"},{"location":"public/#Basis","page":"Public API","title":"Basis","text":"","category":"section"},{"location":"public/","page":"Public API","title":"Public API","text":"TensorBasis\nNonTensorBasis\nTensorH1LagrangeBasis","category":"page"},{"location":"public/#LFAToolkit.TensorBasis","page":"Public API","title":"LFAToolkit.TensorBasis","text":"TensorBasis(\n    p1d,\n    q1d,\n    dimension,\n    nodes1d,\n    quadraturepoints1d,\n    quadratureweights1d,\n    interpolation1d,\n    gradient1d\n)\n\nTensor product basis\n\nArguments:\n\np1d:                 number of nodes in 1 dimension\nq1d:                 number of quadrature points in 1 dimension\ndimension:           dimension of the basis\nnodes1d:             coordinates of the nodes in 1 dimension\nquadraturepoints1d:  coordinates of the quadrature points in 1 dimension\nquadratureweights1d: quadrature weights in 1 dimension\ninterpolation1d:     interpolation matrix from nodes to quadrature points in 1 dimension\ngradient1d:          gradient matrix from nodes to quadrature points in 1 dimension\n\nReturns:\n\nTensor product basis object\n\n\n\n\n\n","category":"type"},{"location":"public/#LFAToolkit.NonTensorBasis","page":"Public API","title":"LFAToolkit.NonTensorBasis","text":"NotTensorBasis(\n    p,\n    q,\n    dimension,\n    nodes,\n    quadraturepoints,\n    quadratureweights,\n    interpolation,\n    gradient\n)\n\nNon-tensor basis\n\nArguments:\n\np:                 number of nodes \nq:                 number of quadrature points\ndimension:         dimension of the basis\nnodes:             coordinates of the nodes\nquadraturepoints:  coordinates of the quadrature points\nquadratureweights: quadrature weights\ninterpolation:     interpolation matrix from nodes to quadrature points\ngradient:          gradient matrix from nodes to quadrature points\n\nReturns:\n\nNon-tensor product basis object\n\n\n\n\n\n","category":"type"},{"location":"public/#LFAToolkit.TensorH1LagrangeBasis","page":"Public API","title":"LFAToolkit.TensorH1LagrangeBasis","text":"TensorH1LagrangeBasis(p1d, q1d, dimension)\n\nTensor product basis on Gauss-Lobatto points with Gauss-Legendre quadrature\n\nArguments:\n\np1d:       number of Gauss-Lobatto nodes\nq1d:       number of Gauss-Legendre quadrature points\ndimension: dimension of basis\n\nReturns:\n\nH1 Lagrange tensor product basis object\n\nExample:\n\n# generate H1 Lagrange tensor product basis\nbasis = TensorH1LagrangeBasis(4, 3, 2);\n\n# verify\n@assert basis.p1d == 4\n@assert basis.q1d == 3\n@assert basis.dimension == 2\n\n# output\n\n\n\n\n\n\n","category":"function"},{"location":"public/#Basis-Evaluation-Mode","page":"Public API","title":"Basis Evaluation Mode","text":"","category":"section"},{"location":"public/","page":"Public API","title":"Public API","text":"EvaluationMode.EvalMode","category":"page"},{"location":"public/#LFAToolkit.EvaluationMode.EvalMode","page":"Public API","title":"LFAToolkit.EvaluationMode.EvalMode","text":"Basis evaluation mode for operator inputs and outputs\n\nModes:\n\ninterpolation:     values interpolated to quadrature points\ngradient:          derivatives evaluated at quadrature points\nquadratureweights: quadrature weights\n\nExample:\n\nEvaluationMode.EvalMode\n\n# output\nEnum LFAToolkit.EvaluationMode.EvalMode:\ninterpolation = 0\ngradient = 1\nquadratureweights = 2\n\n\n\n\n\n","category":"type"},{"location":"public/#Operator-Field","page":"Public API","title":"Operator Field","text":"","category":"section"},{"location":"public/","page":"Public API","title":"Public API","text":"OperatorField","category":"page"},{"location":"public/#LFAToolkit.OperatorField","page":"Public API","title":"LFAToolkit.OperatorField","text":"OperatorField(\n    basis,\n    evaluationmodes\n)\n\nFinite Element operator input or output, with a basis and evaluation mode\n\nArguments:\n\nbasis:           finite element basis for the field\nevaluationmodes: array of basis evaluation modes,                    note that quadrature weights must be listed in a separate operator field\n\nReturns:\n\nFinite element operator field object\n\n\n\n\n\n","category":"type"},{"location":"public/#Operator","page":"Public API","title":"Operator","text":"","category":"section"},{"location":"public/","page":"Public API","title":"Public API","text":"Operator","category":"page"},{"location":"public/#LFAToolkit.Operator","page":"Public API","title":"LFAToolkit.Operator","text":"Operator(\n    weakform,\n    mesh,\n    inputs,\n    outputs\n)\n\nFinite element operator comprising of a weak form and bases\n\nArguments:\n\nweakform: user provided function that represents weak form at quadrature points\nmesh:     mesh object with deformation in each dimension\ninputs:   array of operator input fields\noutputs:  array of operator output fields\n\nReturns:\n\nFinite element operator object\n\n\n\n\n\n","category":"type"},{"location":"background/#Mathematical-Background","page":"Mathematical Background","title":"Mathematical Background","text":"","category":"section"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"Local Fourier Analysis was first used by Brandt to analyze the convergence of multi-level adaptive techniques, but the technique has been adapted for multi-level and multi-grid techniques more broadly.","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"By way of example, we will explore Local Fourier Analysis with the diffusion operator.","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"Consider the PDE","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"- nabla^2 u = f","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"with corresponding weak form","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"int_Omega nabla u cdot nabla v - int_partial Omega nabla u v = int_Omega f v forall v in V","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"for some suitable V subseteq H_0^1 left( Omega right).","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"In Local Fourier Analysis, we focus on single elements or macro-element patches, neglecting the boundary conditions by assuming the boundary is distant from the local element under consideration.","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"a left( u v right) = int_Omega nabla u cdot nabla v","category":"page"},{"location":"background/","page":"Mathematical Background","title":"Mathematical Background","text":"In the specific case of a one dimensional mesh with cubic Lagrage basis on the Gauss-Lobatto points, the assembled stiffness matrix is given by INSERT STIFFNESS MATRIX HERE","category":"page"},{"location":"#LFAToolkit","page":"Introduction","title":"LFAToolkit","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Local Fourier Analysis for arbitrary order finite element type operators","category":"page"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Local Fourier Analysis is a tool commonly used in the analysis of multigrid and multilevel algorthms for solving partial differential equations via finite element or finite difference methods. This analysis can be used to predict convergance rates and optimize parameters in multilevel methods and preconditioners.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"This package provides a toolkit for analyzing the performance of preconditioners for arbitrary, user provided weak forms of partial differential equations.","category":"page"},{"location":"#Contents","page":"Introduction","title":"Contents","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Pages = [\n    \"background.md\",\n    \"public.md\",\n    \"private.md\",\n    \"references.md\"\n]","category":"page"}]
}
