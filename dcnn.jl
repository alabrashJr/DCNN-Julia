using Pkg;Pkg.update(); for p in ("Embeddings","DataFrames","DataStructures","DataFrames","FileIO","LinearAlgebra","Knet","FileIO"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using DataStructures,DataFrames,FileIO,Embeddings,LinearAlgebra,DataFrames;
using Base.Iterators: flatten
using Statistics: mean
using Knet: Knet, conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, Data,minibatch;
using Dates

bels=["abbreviation", "entity", "description", "location" ,"numeric"," "]
revs, W, W2, word_idx_map, vocab=load("Data/TREC_sib.jld2","datas"); 
word_idx_map["ROOT"]=size(W,1);
W=W';

#Transforms sentence into a list of indices. Pad with zeroes.
function get_text_mat(t,word_idx_map;max_l=56,filter_h=5)
    #t the text of question
    x=[] # output matrix
    pad=filter_h -1 # padding number
    for i in collect(1:pad);push!(x,0);end #adding padding 
    words=split(t)
    #extract the unique id of words in the question text and adding it to the matrix 
    for w in words
        if w in keys(word_idx_map);push!(x,word_idx_map[w])
        else; @show w ;end
    end    

    while length(x)<max_l+2*pad    # accomplish 64 +1 size by adding zeros till finish 
            push!(x,0)
    end
    
    return  x
end

function getSen(vector)
#labels=["abbreviation","numeric",  "description", "human","location" ,"entity"]
t=Array{Int}(vector)
println(permutedims(t))
for i in t
    if i==1557;print("?\n y =",t[end]+1 );break;end
    if i==0;continue;end
    for (key,value) in word_idx_map
        if value==i; print(key," ");end
    end
end
end

#Transforms sentence into a list of indices. Pad with zeroes. 
function get_tree_rep(r,word_idx_map)
# question 
#@show t=r["tree"] #the tree of question
    each_sent=deepcopy(r)# output matrix
    for (j, each_word) in enumerate(each_sent[1:end-1])
        #@show (j, each_word)
            for (l, each_field) in enumerate(each_word)
           # @show (l, each_field)
                if each_field in keys( word_idx_map)
                #@show j,l ;
                    each_sent[j]=Array{Any,1}(each_sent[j])
                     each_sent[j][l] = word_idx_map[each_field]
                elseif each_field == 0
                    continue
                else
                    @show each_field
                end
            end
    end       
    return each_sent;
end

function train_dev_test(revs)
    s1,s2,s3=[],[],[]
    t1,t2,t3=[],[],[]
    for rev in revs
    sent =get_text_mat(rev["text"], word_idx_map)   
    push!(sent,rev["y"])
    sent_tensor = get_tree_rep(rev["tree"], word_idx_map)
        
    if rev["split"]==1
            push!(s1,Array{Int}(sent))
            push!(t1,sent_tensor)
    elseif rev["split"]==2
            push!(s2,Array{Int}(sent))
            push!(t2,sent_tensor)
    elseif rev["split"]==3
            push!(s3,Array{Int}(sent))
            push!(t3,sent_tensor)
    end
end

    train = hcat([f1 for f1 in s1]...)
    test =hcat([f1 for f1 in s2]...)
    dev = hcat([f1 for f1 in s3]...)
    train_tensor = t1
    test_tensor = t2
    dev_tensor = t3
    return (train,test,dev),(train_tensor,test_tensor,dev_tensor)
end
dataset,datasetTensor=train_dev_test(revs);

sent1=vcat([permutedims(vcat(datasetTensor[1][x][1:end-1]...)) for x in 1:size(datasetTensor[1],1)]);#sent1=vcat([permutedims(vcat(datasetTensor[1][1][1:end-1]...)) for x in 1:size(datasetTensor[1],1)]...)
sent2=vcat([permutedims(vcat(datasetTensor[2][x][1:end-1]...)) for x in 1:size(datasetTensor[2],1)]);#sent1=vcat([permutedims(vcat(datasetTensor[1][1][1:end-1]...)) for x in 1:size(datasetTensor[1],1)]...);

y_train=Array{Int8}([dataset[1][:,x][end] for x in 1:size(dataset[1],2)]);#ytrainT=[datasetTensor[1][x,:][end][end][end] for x in 1:size(datasetTensor[1],1)];
y_test=Array{Int8}([dataset[2][:,x][end] for x in 1:size(dataset[2],2)]);#ytestT=[datasetTensor[2][x,:][end][end][end] for x in 1:size(datasetTensor[2],1)];

y_train=y_train.+1;
y_test=y_test.+1;

dtrn=minibatch(sent1,y_train,160,shuffle=true);
dtst=minibatch(sent2,y_test,160);

# Let's define a chain of layers

struct Chain
    layers
    Chain(layers...) = new(layers)
end
function (c::Chain)(x)
    #println(Dates.format(now(), "MM:SS"),".Chaindeyim\t",summary(x),"\t",size(x))
    x=KnetArray{Float32}(reshape(W[:, permutedims(hcat(x...))],(300,450,1,160)))
    #println(Dates.format(now(), "MM:SS"),".Chaindeyim\t",summary(x),"\t",size(x))
#     println("\nChaindeyim ,\t ", typeof(x),"\t", summary(x))
    (for l in c.layers; x = l(x); end; x)
end
function (c::Chain)(x,y) 
#     println("\nloss Chaindeyim x ,\t ", typeof(x),"\t", summary(x))
#     println("\nloss Chaindeyim y ,\t ", typeof(y),"\t", summary(y))
    nll(c(x),y)
    
end
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

# Define a convolutional layer:
struct Conv; w; b; f; p;E; end
function (c::Conv)(x)
#     println("\nConvdeyim \t", typeof(x),"\t", summary(x) )
#xx=KnetArray{Float32}(reshape(c.E[:, permutedims(hcat(x...))],(300,450,1,160)))
return c.f.(pool(conv4(c.w, dropout(x,c.p)) .+ c.b))
end
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0,E=W) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop,E)

# Redefine dense layer (See mlp.ipynb):
struct Dense; w; b; f; p; end
function (d::Dense)(x) 
#     println("\nDensedeyim ,\t " , typeof(x),"\t", summary(x))
    d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
end
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

# hidden_units=[100,2] #which meaning ...... how many conv layers 
# dropout_rate=[0.5] #where in which layer ? conv or dense
n_epochs=50;
# batch_size=170, 
lr_decay = 0.005
function trainresults(file,model; o...)
        println("lr =",lr_decay," \t n_epochs= ",n_epochs)
    if (print("Train from scratch? "); readline()[1]=='y')
        takeevery(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)
        r = ((model(dtrn), model(dtst), zeroone(model,dtrn), zeroone(model,dtst))
             for x in takeevery(length(dtrn), progress(sgd(model,repeat(dtrn,n_epochs),lr=lr_decay))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file,"results",r)
        Knet.gc() # To save gpu memory
    else
        if isfile(file);r=Knet.load(file,"results");else;println("there is no file such this");return;end
    end
    println(minimum(r,dims=2))
    return r
end

dcnn5 =   Chain( Conv(3,3,1,5),
Conv(4,4,5,10),
Conv(5,5,10,15),
Dense(27030,1100,pdrop=0.5),
Dense(1100,6,pdrop=0.5))

summary.(l.w for l in dcnn5.layers)

n_epochs=350;
lr_decay = 0.005
cnn9=trainresults("models/dcnn5.jld2", dcnn5);

