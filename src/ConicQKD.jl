module ConicQKD

using Hypatia
using LinearAlgebra
using GenericLinearAlgebra

import Hypatia.RealOrComplex
import Hypatia.Optimizer
using Hypatia.Cones
import Hypatia.Cones.Cone

#these functions are imported in order to extend them
import Hypatia.Cones:
    reset_data,
    use_sqrt_hess_oracles,
    setup_extra_data!,
    get_nu,
    set_initial_point!,
    update_feas,
    update_grad,
    update_hess_aux,
    inv_hess_prod!,
    hess_prod!,
    use_dder3,
    update_dder3_aux,
    dder3

#these are just to be used
import Hypatia.Cones:
    svec_side, svec_to_smat!, smat_to_svec!, spectral_outer!, Δ2!, Δ3!, eig_dot_kron!, alloc_hess!, symm_kron!, is_feas

import MathOptInterface
const MOI = MathOptInterface
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction
const VV = MOI.VectorOfVariables
const VAF = MOI.VectorAffineFunction

include("epiqkdtri.jl")

"""
    EpiQKDTriCone{T,R}(Gkraus::VecOrMat, Zkraus::Vector, dim::Int; blocks::Vector, use_dual::Bool)

QKD cone with number of real parameters `dim`. The cone is parametrized by the CP maps G and Z, given as vectors of Kraus operators `Gkraus` and `Zkraus`. `blocks` is an optional argument describing the block structure of Z as a vector of vectors. `use_dual` is an optional argument to optimize over the dual cone instead.
"""
struct EpiQKDTriCone{T<:Real,R<:RealOrComplex{T}} <: MOI.AbstractVectorSet
    Gkraus::Vector{<:AbstractMatrix}
    Zkraus::Vector{<:AbstractMatrix}
    dim::Int
    blocks::Vector{<:AbstractVector}
    use_dual::Bool

    function EpiQKDTriCone{T,R}(
        Gkraus::Vector{<:AbstractMatrix},
        Zkraus::Vector{<:AbstractMatrix},
        dim::Int;
        blocks::Vector{<:AbstractVector} = [1:size(Zkraus[1], 1)],
        use_dual::Bool = false
    ) where {T<:Real,R<:RealOrComplex{T}}
        new{T,R}(Gkraus, Zkraus, dim, blocks, use_dual)
    end
end
export EpiQKDTriCone

MOI.dimension(cone::EpiQKDTriCone) = cone.dim

function Hypatia.cone_from_moi(::Type{T}, cone::EpiQKDTriCone{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    return EpiQKDTri{T,R}(cone.Gkraus, cone.Zkraus, cone.dim; blocks = cone.blocks, use_dual = cone.use_dual)
end

const NewCones{T<:Real} = Union{EpiQKDTriCone{T,T},EpiQKDTriCone{T,Complex{T}}}

const NewSupportedCone{T<:Real} = Union{Hypatia.SupportedCone{T},NewCones{T}}

Base.copy(cone::NewCones) = cone

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:Union{VV,VAF{T}},<:NewSupportedCone{T}}
) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    z_i = opt.solver.result.z[opt.moi_cone_idxs[i]]
    return untransform_affine(opt.moi_cones[i], z_i)
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:Union{VV,VAF{T}},<:NewSupportedCone{T}}
) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    s_i = opt.solver.result.s[opt.moi_cone_idxs[i]]
    return untransform_affine(opt.moi_cones[i], s_i)
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{<:Union{VV,VAF{T}}},
    ::Type{<:Union{MOI.Zeros,NewSupportedCone{T}}}
) where {T<:Real}
    return true
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex{VAF{T},<:NewSupportedCone{T}},
    chg::MOI.VectorConstantChange{T}
) where {T}
    i = ci.value
    idxs = opt.moi_cone_idxs[i]
    set = opt.moi_cones[i]
    new_h = chg.new_constant
    if needs_rescale(set)
        rescale_affine(set, new_h)
    end
    if needs_permute(set)
        new_h = h[permute_idxs(set)]
    end
    Solvers.modify_h(opt.solver, idxs, new_h)
    return
end

end # module ConicQKD
