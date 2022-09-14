include("myMPSstuff.jl")
include("myMPOstuff.jl")
include("myMPSMPOstuff.jl")

function main()
    L = 20
    gg = 0.9
    mps1 = random_MPS(L)
    println(get_norm_zip(mps1))
    bring_canonical!(mps1, 50)
    println(get_norm_zip(mps1))

    Hi=build_Ising_MPO(L, gg)
    Htr = expMinusEpsHIsingMPO(L, gg)


    psiGS = power_method(Hi, 600, 50)

    println(get_entropies(psiGS))

    println(expval_MPO(psiGS, Hi))


    psiGS = power_method(Htr, 600, 50)
    println(get_entropies(psiGS))

    println(expval_MPO(psiGS, Hi))

end

main()
