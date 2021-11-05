#!/bin/bash
#cd ../
#julia --project -e 'using Pkg; Pkg.rm(PackageSpec(path="LFAToolkit.jl"))'
#cd LFAToolkit.jl
julia --project -e 'using Pkg; Pkg.build()'
cd ../
julia --project -e 'using Pkg; Pkg.add(PackageSpec(path="LFAToolkit.jl"))'
cd LFAToolkit.jl
