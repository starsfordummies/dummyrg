include("myMPSstuff.jl")
include("myMPOstuff.jl")
include("myMPSMPOstuff.jl")

using ..myMPSstuff: get_entropies, bring_canonical!
using ..myMPOstuff: expMinusEpsHIsingMPO_compact, expMinusEpsHIsingMPO
#using ..myMPsMPOstuff: power_method, expval_MPO

function main()
    L = 100
    gg = 0.9

    #Htr = expMinusEpsHIsingMPO_compact(L, gg)
    Htr = expMinusEpsHIsingMPO(L, gg)
    
    psiGS = power_method(Htr, 2, 50, false)
    psiGS = power_method(Htr, 2, 50, true)


    @time  psiGS = power_method(Htr, 600, 200)
    println(get_entropies(psiGS))
    #println(expval_MPO(psiGS, Hi))
    
    @time  psiGS = power_method(Htr, 600, 200, true)

    println(get_entropies(psiGS))
    #println(expval_MPO(psiGS, Hi))

end

main()
