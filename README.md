# DCNN-Julia

A final project for Comp 541 Deep Learning Course By Deniz Yuret. 
This project will be replicated [Dependency-based Convolutional Neural Networks for Sentence Embedding](http://people.oregonstate.edu/~mam/pdf/papers/DCNN.pdf) results using Knet.

models accuracies : https://drive.google.com/file/d/13azo5E8JSoapbbLf6mXIBG3kfKqh7rnZ/view?usp=sharing

Setup : https://docs.google.com/document/d/1lyTPLovvrCkhU1NIUpxNmgOyKWNYgQzCQG1Cdv-tveg/

Research Log: https://docs.google.com/spreadsheets/d/1yE3PwbQjsSqPG3YOr9sSb0x6-JGEXnobi3ct-RZUu2w/

persentation: https://docs.google.com/document/d/1lyTPLovvrCkhU1NIUpxNmgOyKWNYgQzCQG1Cdv-tveg/

Datasheet: https://docs.google.com/spreadsheets/d/1vkZVL3cdjIBhrTeOTqbaTDuQXeHW_J6rP7MCSOOSj3A


![model_structure](https://user-images.githubusercontent.com/9295206/58470550-98a4a200-814a-11e9-8e01-e9288a0fa42b.jpg)


![alt text](https://raw.githubusercontent.com/alabrashjr/DCNN-Julia/master/Loss.png)



![alt text](https://raw.githubusercontent.com/alabrashjr/DCNN-Julia/master/Error.png)


```

git clone https://github.com/alabrashJr/DCNN-Julia
cd DCNN-Julia && git clone https://github.com/chentinghao/download_google_drive.git
pip install tqdm
python download_google_drive/download_gdrive.py 1YWl_p2FtzOWhknwVxAzFD5740hnUhwe0 ./google_w2v.bin
python download_google_drive/download_gdrive.py 1cUpzjy9ASH1pJQeJkM1KoKMcdBwe9kDT  ./Data.zip
unzip Data.zip -d ./
rm Data.zip

```
