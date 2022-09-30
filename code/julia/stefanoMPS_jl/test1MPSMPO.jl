using myMPSmod

#get_entropies, bring_canonical!, get_norm_zip,
#               expMinusEpsHIsingMPO_compact, expMinusEpsHIsingMPO
#using ..myMPsMPOstuff: power_method, expval_MPO

using BenchmarkTools

function main()
    L = 100
    gg = 0.9

    #Htr = expMinusEpsHIsingMPO_compact(L, gg)
    Htr = expMinusEpsHIsingMPO(L, gg)
    
    psiGS = power_method(Htr, 2, 50, false)
    psiGS = power_method(Htr, 2, 50, true)


    psiGS = power_method(Htr, 50, 100)
    println(get_entropies(psiGS))
    #@btime  psiGS = power_method($Htr, 50, 100)
    #println(expval_MPO(psiGS, Hi))
    
    psiGS = power_method(Htr, 50, 100, true)
    println(get_entropies(psiGS))
    #@btime power_method($Htr, 50, 100, true)

    #println(get_entropies(psiGS))
    #println(expval_MPO(psiGS, Hi))

end

main()
