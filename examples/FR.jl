using Ket
using LinearAlgebra

"Kraus operator for the key map"
function gkrausTop(pK::T) where {T<:AbstractFloat} 
    QB_Z = proj(1,3)+proj(2,3) 
    G = sqrt(pK)*sum([kron(ket(i,3),kron(proj(i),QB_Z)) for i=1:2])
    return  G
end  

# choose a pK
pK = 0.95

G_top = gkrausTop(pK)
gg = G_top*G_top'
λ,P = eigen(gg)
W = P[:,15:18]
Ghat = W'*G_top


Z_top = [kron(proj(i,3),I(6)) for i=1:2]
ZG_top = [Zi*G_top for Zi in Z_top]
ZGhat = Vector{Matrix{Float64}}(undef,2)
for i=1:2
    λ,U = eigen(ZG_top[i]*ZG_top[i]')
    idx_nonzero = findall(>(1e-10), λ)
    ZGhat[i]=U[:,idx_nonzero]'*ZG_top[i]
end
ZGhat