#!/bin/bash
julia --project -e 'using Pkg; Pkg.build()'
cd ../
julia --project -e 'using Pkg; Pkg.add(PackageSpec(path="LFAToolkit.jl"))'
cd LFAToolkit.jl
