module JuMPExt

import ConicQKD._numerical_type
import JuMP

_numerical_type(::Type{T}) where {T<:JuMP.AbstractJuMPScalar} = JuMP.value_type(T)

end # module
