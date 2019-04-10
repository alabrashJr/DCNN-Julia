# @Abdulrahman Alabrash 
# https://github.com/alabrashJr/DCNN-Julia
# 
# Prepare the data for DCN
#     * build_data_cv :taking The data and its labels in addition to split dictionary which determine which tuple is train or test data, returns, 
#                 * revs, which is a list of datum  = {"y": 
#                                     "text":,                             
#                                     "num_words": ,
#                                     "split": 
#                              }
#                 * Voca defaultdict which indicates the number of each word occurrence 
# 
#     * sibling2, take sentence and return a sibling list for each word sibling dependency list. The list elements will be as following.
#             * [1] Word 
#             * [2] Parent 
#             * [3] sibling1: 
#                     * If word index < parent index:
#                             *  look for sibling indices < word index 
#                                     * If there are not add  “Start”
#                     * If word index > parent index:
#                             * Look for sibling indices>word index 
#                                     * If there are not add  “Stop”
#             * [4] Sibling2:
#                     *  If word index < parent index:
#                             *  look for sibling indices < word index 
#                                     * If there are not add  “Start”
#                     * If word index > parent index:
#                             * Look for sibling indices>word index 
#                                     * If there are not add  “Stop”
#             * [5] grand parent: 
#                     * If is available add it if not add “Root”
# 
# 
#     * set_sibling2, execute sibling2 method for each sentence and padding it to the maxl which is 45, and add the label of the sentence as a list so the final length will be 46, for each sentence: size(#sentence(maxl(5))
# 
# 
#     * set_conv_sent, extract the 4 ancestors of the word, padding it to the maxl which is 45,and add the label of sentence as list so the final length will be 46, the returned value will be equal to header list + following list for each word In each sentence : size(#sentence(maxl(5))
#         * For each sentence 
#         * [5 x Root]
#         * [4x Root,1st Word]
#         * [3xRoot,1stWord,1st ancestor]
#         * [2xRoot,1stWord,1st:2rd ancestor]
#         * [Root,1stWord,1st:3rd ancestor]
#         * For each word 
#             * [ Word, 1st ancestor, 2nd ancestor, 3rd ancestor, 4th ancestor]
# 
# 
# * revs= Dict{String,Any} with 5 entries:
#         y-> label of the questions 1-5
#         num_words-> length of questions
#         tree -> concrete  the ancestors array with siblings array -> length of output array will be(#sentence(maxl(5+5))
#         text -> the question text
#         split -> type of tuple (training, test , div) 
# 
# 
# * W = word embedding using google2vec  size=10097×300
# 
# * W2= word emeding using uniform dist between -0,25 <-> 0,25    size=10097×300
# 
# * word_idx_map= word indices in W matrices len=10097
# 
# * vocab= vocab defalut Dic {word,number of occurence}  len=10097
# 
# [revs, W, W2, word_idx_map, vocab] -> TREC_sib.jld2

using Pkg;Pkg.update()
for p in ("Embeddings","DataStructures","DataFrames","FileIO","LinearAlgebra","FileIO"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using DataStructures,DataFrames,FileIO,Embeddings,LinearAlgebra,FileIO;

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

function get_split(size=5953)
dict=Dict()
    for i in range(1,length=size)
        if i < 5453
            dict[i] = 1
        else
            dict[i] =2
        end 
    end 
    return dict
end

function get_labels(fn)
    f=open(fn,"r")
    dict=Dict()
    for (index, i) in enumerate(readlines(f))
        dict[index] = parse(Int,i) 
    end
    return dict
end

function clean_str(string, TREC=false)
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    
    string = replace(string,r"[^A-Za-z0-9(),!?\'\`]" =>s" ")
    string = replace(string,r"\'s" =>s" 's") 
    string = replace(string,r"\'ve" =>s" 've") 
    string = replace(string,r"n\'t" =>s" n't") 
    string = replace(string,r"\'re" =>s" 're") 
    string = replace(string,r"\'d" =>s" 'd") 
    string = replace(string,r"\'ll" =>s" 'll") 
    string = replace(string,r"," =>s" , ") 
    string = replace(string,r"!" =>s" ! ") 
    string = replace(string,r"\(" =>s" \\( ") 
    string = replace(string,r"\)" =>s" \\) ") 
    string = replace(string,r"\?" =>s" \\? ") 
    string = replace(string,r"\s{2,}" =>s" ")    
        
    return (TREC ?  strip(string) : lowercase(strip(string)))
end

# revs: Creats datum Dictionary which contains each data tuple's label,question text, length of question text,type of tuple(train-test-dev)
# vocab: default Dict ,counts counts the words occusions through all tuples
function build_data_cv(file, split_dict, label_dict, clean_string=false)
    """
    Loads data and split data
    """
    revs = []
    f = open(file,"r")
    vocab = DefaultDict(0)#https://juliacollections.github.io/DataStructures.jl/latest/default_dict.html
    
    for (index, line) in enumerate(readlines(f))     
        rev = []
        push!(rev,strip(line))
        if clean_string
            orig_rev = clean_str(join(rev," "))
        else
            orig_rev = join(rev," ")
        end
        words = Set(split(orig_rev))
        for word in words
            vocab[word] += 1
        end
        datum  = Dict("y"=>label_dict[index], 
                    "text"=> orig_rev,                             
                    "num_words"=> length(split(orig_rev)),
                    "split"=> split_dict[index])
        push!(revs,datum)
    end

    return revs, vocab
end

##converting each object of the tree to structured list of trees. 
#<img width="657" alt="Screen Shot 2019-04-02 at 15 45 50" src="https://user-images.githubusercontent.com/9295206/55403412-77ff0800-555e-11e9-9cc7-e8977c06cb31.png">  
function sibling2(sents, opt)
    sent_list = []
    kez= sort(collect(keys(sents)))
    for key in kez
        currnet_node=sents[key]
        if key == 0;continue;end
        #currnet_node = sents[key]
        word_list = []
        push!(word_list,currnet_node.word)
        
        parent_index = currnet_node.parentindex
        parent = sents[parent_index]
        push!(word_list,parent.word)
        sib_list = parent.kidsindex
        if key < parent_index
            sib_candidate = [i for i in sib_list if i < key]
            if sib_candidate == [];push!(word_list,"*START*")
            else;push!(word_list,sents[pop!(sib_candidate)].word);end 
            if sib_candidate == [];push!(word_list,"*START*")
            else;push!(word_list,sents[pop!(sib_candidate)].word);end
        else
            sib_candidate = [i for i in sib_list if i > key]
            if sib_candidate == [];push!(word_list,"*STOP*")
            else;push!(word_list,sents[pop!(sib_candidate)].word);end
            if sib_candidate == [];push!(word_list,"*STOP*")
            else;push!(word_list,sents[pop!(sib_candidate)].word); end
       end
        grad_parent_ind = parent.parentindex
        grad_word = sents[grad_parent_ind].word
        push!(word_list,grad_word)
        push!(sent_list,word_list)
    end
    return sent_list
 end
                                                    
 #creats lists of strcutred trees + padding till max =45 + label list 
function set_sibling2(tree,labels_dict,max_len)

    sent_num = length(tree)
    doc_list =[]
    for (ind,sents) in enumerate(tree)
        sib_6 = sibling2(sents,6)
        sent_list = sib_6
        dummy_len = length(sent_list[1])
        dummy = repeat(["*ZERO*"],dummy_len)
        while length(sent_list) < max_len #padding tree to the maximum tree by adding zeros list to sent_lists
            push!(sent_list,dummy)
        end
        currnet_label = labels_dict[ind]
        class_dummy = repeat([currnet_label],dummy_len)
        push!(sent_list,class_dummy)        
        push!(doc_list,sent_list)# adding the list
        
    end
    return doc_list    
end

function set_conv_sent(tree,labels_dict,max_len)
    conv_length = 5
    @show sent_num = length(tree)
    ##65-4 the most beginning 4 will be append to the front at last
    #sent_tensor = np.array.zeros((1,61,5))
    #sent_counter = 0
    doc_list =[]
    for (ind,sents) in enumerate(tree)
         sent_list = []
        kez= sort(collect(keys(sents)))
        for key in kez
            #@show key 
            if key == 0;continue;end
            currnet_node = sents[key]
            word_list = []
            for i in range(1,conv_length)
                #@show currnet_node.word
                if currnet_node.word != "ROOT";push!(word_list,currnet_node.word)
                else; push!(word_list,currnet_node.word);end
                if currnet_node.word != "ROOT"; currnet_node = sents[currnet_node.parentindex];end
            end
             push!(sent_list,word_list)
            #@show length(sent_list)
        end 
        header = []
        dummy = repeat(["ROOT"],conv_length)
        for i in range(1,conv_length-1);push!(header,vcat(dummy[1:conv_length-i], sent_list[1][1:i]));end
        sent_list = vcat(header,sent_list)
        while length(sent_list) < max_len;push!(sent_list,dummy);end
        currnet_label = labels_dict[ind]
        class_dummy = repeat([currnet_label],conv_length)
        push!(sent_list,class_dummy)
        #@show length(sent_list)
        push!(doc_list,sent_list)
            end     
   # @show length(doc_list)
    return doc_list
end

function add_tree2vocab(sent, vocab)
    
    for (j, each_word) in enumerate(sent[1:end-1])
        for (l, each_field) in enumerate(each_word)
            if each_field in keys(vocab);continue
            elseif each_field == 0;continue
            elseif each_field == "ROOT";continue
                else;vocab[each_field] += 1;end
        end
    end
end

function merge_two(revs, tree)
    counter=1
    for i in revs
        sent2 = tree[counter]
        counter += 1
        i["tree"] = sent2
    end    
    return revs
        
end

function load_bin_vec(fname, vocab)
pf(s)=return parse(Int,s)
pc(s)=return convert(Char,s[1])
word_vecs = Dict()
    open(fname, "r") do f
                @show header = readline(f)
                vocab_size, layer1_size = map(pf, split(header))
                @show binary_len = sizeof(Float32) * layer1_size
                #@show  binary_len = layer1_size
             for line in collect(1:vocab_size)
                word=[]
                while true 
                           ch=read(f,1)
                           ch=convert(Char,ch[1])
                            if ch == ' '
                                word = join(word,"")
                                break
                            end
                            if ch != '\n';
                                push!(word,ch);
                            end
                    end
                if word in keys(vocab)
                   word_vecs[word]=reinterpret(Float32,read(f,binary_len))
                else
                read(f,binary_len)
                end

            end    
    end;
return word_vecs
end

function add_unknown_words(word_vecs, vocab, min_df=1, k=300)
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for (word,w) in vocab
        if word ∉ keys(word_vecs) && vocab[word] >= min_df
            word_vecs[word] = (rand(k).*0.5).- 0.25
        end
    end
end

function get_W(word_vecs, k=300)
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = length(word_vecs)
    word_idx_map = Dict()
    W = zeros((vocab_size+1, k))            
    W[1,:] =  zeros(300)
    i = 1
    for (word,w) in word_vecs
        W[i,:] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    end
    return W, word_idx_map
    end

w2v_file = "google_w2v.bin"   
sent_file = "Data/TREC_all.txt"
#tree_file = "Data/data.jld2" # hdf5 wrtoe 
label_file = "Data/label_all.txt"
label_dict = get_labels(label_file);
split_dict = get_split(5952) ;
       
revs, vocab = build_data_cv(sent_file, split_dict, label_dict); 
function dfun(d::Dict);return d["num_words"];end
max_l,maxIndex = findmax(map(dfun, revs)) # find the longest text length 
    
all_tree = load("Data/data.jld2","data"); # load node objects that have been created in pre-indexing file 
data_sibling = set_sibling2(all_tree,label_dict,max_l+8);
data_tree = set_conv_sent(all_tree,label_dict,max_l+8); 
#summary.(data_tree)

new_data_tree = []
for (ind,l) in enumerate(data_tree)
    new_list=[]
    for (ind2,l2) in enumerate(l);push!(new_list,vcat(data_tree[ind][ind2],data_sibling[ind][ind2]));end
    push!(new_data_tree,new_list)
end
data_tree = new_data_tree
#@show length.(new_data_tree)
for i in data_tree;add_tree2vocab(i, vocab);end
@show length(vocab)
revs = merge_two(revs,data_tree);

println("number of sentences: ", length(revs))
println("vocab size: " ,length(vocab))
println("max sentence length: " ,max_l+8)
println("loading word2vec vectors...")
w2v = load_bin_vec(w2v_file, vocab)
#w2v=Dict()
println("word2vec loaded!")
println("num words already in word2vec: ",length(w2v))
vocab["ROOT"]=1
vocab["*START*"]=1
vocab["*STOP*"]=1
vocab["*ZERO*"]=1   
add_unknown_words(w2v, vocab)
println("num words already in word2vec: ",length(w2v))
W, word_idx_map = get_W(w2v)
rand_vecs = Dict()
add_unknown_words(rand_vecs, vocab)
W2, _ = get_W(rand_vecs)
save("Data/TREC_sib.jld2","datas",[revs, W, W2, word_idx_map, vocab])  
println("dataset created!")

