module myMPSmod

include("myMPSstuff.jl")
include("myMPOstuff.jl")
include("myMPSMPOstuff.jl")


export myMPS, init_MPS, random_mps,
       truncate_svd,  
       bring_canonical!, bring_canonical_opt!,
       overlap, get_norm_zip, check_norm_SVs, svd_sweep!,
       get_entropies

export myMPO, myMPOcompact,
       build_Ising_MPO_compact,
       expMinusEpsHIsingMPO,
       expMinusEpsHIsingMPO_compact


export apply_MPO!, expval_MPO, power_method
       


#using ..myMPSstuff: truncate_svd
#using .myMPSstuff: myMPS, bring_canonical!, bring_canonical_opt!, random_mps, overlap, svd_sweep!
#using .myMPOstuff: myMPO, myMPOcompact, build_Ising_MPO_compact

end # module myMPSmod
