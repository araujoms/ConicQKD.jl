using Hypatia
using LinearAlgebra
using GenericLinearAlgebra

import Hypatia.RealOrComplex
import Hypatia.Optimizer
using Hypatia.Cones
import Hypatia.Cones.Cone

#these functions are imported in order to extend them
import Hypatia.Cones.use_dder3
import Hypatia.Cones.reset_data
import Hypatia.Cones.setup_extra_data!
import Hypatia.Cones.get_nu
import Hypatia.Cones.set_initial_point!
import Hypatia.Cones.barrier
import Hypatia.Cones.update_feas
import Hypatia.Cones.update_grad
import Hypatia.Cones.update_hess

#these are just to be used
import Hypatia.Cones.svec_side
import Hypatia.Cones.svec_to_smat!
import Hypatia.Cones.smat_to_svec!
import Hypatia.Cones.spectral_outer!
import Hypatia.Cones.Δ2!
import Hypatia.Cones.eig_dot_kron!
import Hypatia.Cones.alloc_hess!
import Hypatia.Cones.symm_kron!

using DocStringExtensions

using SparseArrays
import MathOptInterface
const MOI = MathOptInterface
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction
const VV = MOI.VectorOfVariables
const VAF = MOI.VectorAffineFunction

include("epiqkdtri.jl")

"""
Here is the QKD Cone
"""
struct EpiQKDTriCone{T<:Real,R<:RealOrComplex{T}} <: MOI.AbstractVectorSet
    Gkraus::Vector
    Zkraus::Vector
    dim::Int
    use_dual::Bool
end
export EpiQKDTriCone

function EpiQKDTriCone{T,R}(Gkraus::Vector, Zkraus::Vector, dim::Int) where {T<:Real,R<:RealOrComplex{T}}
    return EpiQKDTriCone{T,R}(Gkraus, Zkraus, dim, false)
end

MOI.dimension(cone::EpiQKDTriCone) = cone.dim

function Hypatia.cone_from_moi(::Type{T}, cone::EpiQKDTriCone{T,R}) where {T<:Real,R<:RealOrComplex{T}}
    return EpiQKDTri{T,R}(cone.Gkraus, cone.Zkraus, cone.dim; use_dual = cone.use_dual)
end

const NewCones{T<:Real} = Union{EpiQKDTriCone{T,T},EpiQKDTriCone{T,Complex{T}}}

const NewSupportedCone{T<:Real} = Union{Hypatia.SupportedCone{T},NewCones{T}}

Base.copy(cone::NewCones) = cone

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{<:Union{VV,VAF{T}},<:NewSupportedCone{T}}) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    z_i = opt.solver.result.z[opt.moi_cone_idxs[i]]
    return untransform_affine(opt.moi_cones[i], z_i)
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{<:Union{VV,VAF{T}},<:NewSupportedCone{T}}) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    s_i = opt.solver.result.s[opt.moi_cone_idxs[i]]
    return untransform_affine(opt.moi_cones[i], s_i)
end

function MOI.supports_constraint(::Optimizer{T}, ::Type{<:Union{VV,VAF{T}}}, ::Type{<:Union{MOI.Zeros,NewSupportedCone{T}}}) where {T<:Real}
    return true
end

function MOI.modify(opt::Optimizer{T}, ci::MOI.ConstraintIndex{VAF{T},<:NewSupportedCone{T}}, chg::MOI.VectorConstantChange{T}) where {T}
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
