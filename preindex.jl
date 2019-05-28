# @Abdulrahman Alabrash 
# https://github.com/alabrashJr/DCNN-Julia
#Parse each line trec_all_parsed.txt file to node struct

using Pkg; for p in ("JLD2","FileIO"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using JLD2,FileIO

mutable struct node 
    word
    kidsword
    kidsindex
    parent
    finished
    is_word
    selfindex
    parentindex
    label
    ind
node(word) = word == nothing ? new(nothing,nothing,nothing,nothing,nothing,0,nothing,nothing,nothing,nothing) : new(word,[],[],[],0,1,0,0,"",-1)
    
end

input_file  = "Data/TREC_all_parsed.txt";

input = open(input_file,"r")
counter =0
doc = []
fileReaded=readlines(input)
for i in fileReaded
           counter += 1
            #counter
           #i = strip(i)
           i = strip(i,['[',']',' '])
           #print i
           current = split(i,"), ")
           node_container = Dict()
           ROOT = node("ROOT")
           node_container[0] = ROOT    
           for (index,j) in enumerate(current)
               
               label = split(j,'(')[1]
               j = strip((split(j,'(')[2]),')')
               current_list = [strip(i) for i in split(j,", ")]
               current_node=node(join(split(current_list[2],'-')[1:end-1],'_'))
               current_node.selfindex=parse(Int,split(current_list[2],'-')[end])
               current_node.label = label
               if join(split(current_list[1],'-')[1:end-1],'_') == "ROOT"
                       current_node.parent = ROOT
               else               
                    current_node.parent=join(split(current_list[1],'-')[1:end-1],'_')
                    current_node.parentindex=parse(Int,split(current_list[1],'-')[end])
               end    
               node_container[current_node.selfindex] = current_node
           
           end
           node_container1 = deepcopy(node_container);
            for (index,current) in node_container1
               p_index = current.parentindex
               if p_index ∉ keys(node_container)
                   new_node = node(current.parent)
                   new_node.parent = ROOT
                   new_node.selfindex = p_index
                   node_container[p_index] = new_node    
               end
            end
           
            for (index,current) in node_container1
               if current.word == "ROOT"
                   continue
               end
               p_index = current.parentindex
               #print p_index
               if p_index ∉ keys(node_container)
                   new_node = node(current.parent)
                   new_node.parent = ROOT
                   new_node.selfindex = p_index
                   node_container[p_index] = new_node
               end
               p = node_container[p_index]
               
               push!(p.kidsword,current.word)
               push!(p.kidsindex,current.selfindex)
           end
           push!(doc,node_container)
end

save("Data/data.jld2","data",doc) # using FileIO pacakge 
