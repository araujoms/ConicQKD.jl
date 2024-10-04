# ConicQKD
Implementation of the QKD cone introduced in the paper

### [Quantum key distribution rates from non-symmetric conic optimization](http://arxiv.org/abs/2407.00152)
#### Andrés González Lorente, Pablo V. Parellada, Miguel Castillo-Celeita, and Mateus Araújo

Version 0.2 incorporates an optimized technique to compute the inverse Hessian from the appendix B.2 of [Exploiting Structure in Quantum Relative Entropy Programs](https://arxiv.org/abs/2407.00241), by Kerry He, James Saunderson, and Hamza Fawzi.

## Installation

First you need to install [Julia](https://docs.julialang.org/en/v1/manual/getting-started/). From within Julia, enter the package manager by typing `]`. Then install ConicQKD:
```julia
pkg> add https://github.com/araujoms/ConicQKD.jl
```
This will automatically install all dependencies. The main one is the solver [Hypatia](https://github.com/jump-dev/Hypatia.jl), which this package extends.
## Usage

Several examples are available in the `examples` folder. They are all formulated using the modeller [JuMP](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/). To constraint a quantum state `ρ` to belong to the QKD cone with CP maps `Ghat` and `Zhat` the syntax is

```julia
@constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat, Zhat, 1 + vec_dim; blocks))
```
- `model` is the JuMP optimization model being used
- `h` is a variable which will have the conditional entropy in base e
- `ρ_vec` is a vectorization of `ρ` in the svec format (we provided a function `svec` to compute it).
- `T` is the floating point type to be used (e.g. `Float64`, `Double64`, `Float128`, `BigFloat`, etc.)
- `R` is either equal to `T`, in order to optimize over real matrices, or equal to `Complex{T}` in order to optimize over complex matrices.
- `Ghat` and `Zhat` encode the CP maps as vectors of Kraus operators. 
- `vec_dim` is the number of real parameters of `ρ`, i.e., either d^2 or d(d+1)/2 for the complex and real cases, respectively.
- `blocks` is an optional keyword argument specifying the block structure of `Zhat` as a vector of vectors. For example, if `Zhat` maps a 4x4 `ρ` to a matrix `M` such that only `M[1:2,1:2]` and `M[3:4,3:4]` are nonzero, then `blocks` should be `[1:2, 3:4]`. If this argument is omitted the computation will be considerably slower.
