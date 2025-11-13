# ConicQKD
Implementation of the convex cones introduced in the papers

### Finite-size quantum key distribution rates from Renyi entropies using conic optimization
#### Mariana Navarro, Andrés González Lorente, Pablo V. Parellada, Carlos Pascual-García, and Mateus Araújo

and

### [Quantum key distribution rates from non-symmetric conic optimization](http://arxiv.org/abs/2407.00152)
#### Andrés González Lorente, Pablo V. Parellada, Miguel Castillo-Celeita, and Mateus Araújo

## Installation

First you need to install [Julia](https://docs.julialang.org/en/v1/manual/getting-started/). From within Julia, enter the package manager by typing `]`. Then install ConicQKD:
```julia
pkg> add https://github.com/araujoms/ConicQKD.jl
```
This will automatically install all dependencies. The main one is the solver [Hypatia](https://github.com/jump-dev/Hypatia.jl), which this package extends.
## Usage

Simplified examples that demonstrate how to use the cones are available in the `examples` folder. They can also be used to reproduce the results of "Quantum key distribution rates from non-symmetric conic optimization". To reproduce the results of "Finite-size quantum key distribution rates from Renyi entropies using conic optimization" use the code available in [MarianaNvrr/Renyi-ConicQKD](https://github.com/MarianaNvrr/Renyi-ConicQKD).

The interface is formulated using the modeller [JuMP](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/).

To constraint a quantum state `ρ` to belong to the von Neumann QKD cone with CP maps `Ghat` and `ZGhat` the syntax is
```julia
@constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat, ZGhat, 1 + length(ρ_vec); blocks))
```
To constraint a quantum state `ρ` to belong to the FastRényiQKD cone with CP maps `Ghat`, `ZGhat`, isometry `S`, and Rényi parameter `β = 1/α` the syntax is
```julia
@constraint(model, [h; ρ_vec] in EpiFastRenyiQKDTriCone{T,R}(β, Ghat, ZGhat, 1 + length(ρ_vec); S, blocks)
```
To constraint quantum states `ρ`, `σ` to belong to the RényiQKD cone with CP maps `Ghat`, `Zhat`, isometry `S`, and Rényi parameter `γ = α/(2α - 1)` the syntax is
```julia
@constraint(model, [h; ρ_vec; σ_vec] in EpiRenyiQKDTriCone{T,R}(γ, Ghat, Zhat, 1 + length(ρ_vec) + length(σ_vec); S, blocks))
```
- `model` is the JuMP optimization model being used.
- `h` is a variable which will have the conditional entropy in base e.
- `ρ_vec`, `σ_vec` are vectorizations of `ρ`, `σ` in the svec format (we provided a function `svec` to compute it).
- `T` is the floating point type to be used (e.g. `Float64`, `Double64`, `Float128`, `BigFloat`, etc.).
- `R` is either equal to `T`, in order to optimize over real matrices, or equal to `Complex{T}` in order to optimize over complex matrices.
- `Ghat`, `ZGhat`, and `Zhat` encode the CP maps as vectors of Kraus operators.
- `blocks` is an optional keyword argument specifying the block structure of `Zhat` as a vector of vectors. For example, if `Zhat` maps a 4x4 `ρ` to a matrix `M` such that only `M[1:2,1:2]` and `M[3:4,3:4]` are nonzero, then `blocks` should be `[1:2, 3:4]`. If this argument is omitted the computation will be considerably slower.
- `S` is an optional keyword argument, assumed to be identity if ommitted.
