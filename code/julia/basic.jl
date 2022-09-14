using TensorOperations
using Tullio

m1 = rand(1:20, 3, 7, 10)
m2 = rand(1:20, 7, 2, 10)

@tensor m12te[i,j] := m1[i,a,b]*m2[a,j,b]
@tullio m12tu[i,j] := m1[i,a,b]*m2[a,j,b]


@tensor m12te[i,j] := m1[i,a,b]*m2[a,j,b]
@tullio m12tu[i,j] := m1[i,a,b]*m2[a,j,b]


@time @tensor m12te[i,j] := m1[i,a,b]*m2[a,j,b]
@time @tullio m12tu[i,j] := m1[i,a,b]*m2[a,j,b]

println(length(m1))

println(size(m1)[1])

struct test_struct
    a::Int
    b::String
end

ts1=test_struct(1,"aa")
println(ts1)

Base.@kwdef struct test_struct2
    a::Int
    b::String
end
ts2= test_struct2(a=3,b="bb")

println(ts2)
