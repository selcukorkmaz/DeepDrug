#### Get the dataset ####

aid = "AID_485314" # change AID
data = read.csv(paste0("~/Documents/Studies/DeepDrug/PaDEL-Descriptor/CSV/",aid,"_datatable_all.csv"))
dim(data)

cid = data.frame(data$PUBCHEM_CID[complete.cases(data$PUBCHEM_CID)])
colnames(cid) = "cids"
head(cid)

write.table(cid,paste0("~/Documents/Studies/DeepDrug/PaDEL-Descriptor/cids/",aid,".txt"),
            quote = F, row.names = F)

s = c(seq(0,nrow(cid),120), nrow(cid))
write.table(s, paste0('/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/seq/',aid,'cidssequence.txt'), quote = F, row.names = F)


length(unique(cid[,1]))

length((data$PUBCHEM_ACTIVITY_OUTCOME))
table((data$PUBCHEM_ACTIVITY_OUTCOME))


########### Download SDF files #####

# Note: This is python code

import pubchempy as pm
import pandas as pd

aid = "485314" # change AID
f = 15

cids = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/cids/AID_"+aid+".txt", delimiter="\t")
cids2 = cids.values
cids2 = cids2[:,0]
cids3 = cids2.tolist()
cids4=list(map(int, cids3))

seq = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/seq/AID_"+aid+"cidssequence.txt", delimiter="\t")
seq2 = seq.values
seq2 = seq2[:,0]
seq3 = seq2.tolist()
seq4=list(map(int, seq3))

seq5 = seq4[f:seq.shape[0]]

for i in seq5:
  c = cids4[i:i+120]
f=f+1
path = '/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/SDF/AID_'+aid+'/AID_'+aid+'_'+str(f)+'.sdf'
pm.download('SDF', path, c)
print(f)
print ('%'+str(round((float(f)/seq.shape[0])*float(100))))


#################### Generate descriptors using PADEL ########
aid = "AID_485314" # change AID

setwd('/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/')

f = list.files(paste0('/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/SDF/',aid,'/'))

for(i in 2577:length(f)){
  
  system(paste0('java -jar PaDEL-Descriptor.jar  -threads 4 -2d -3d  -fingerprints -file /Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/Result/',aid,'/',aid,'_',i,'.csv  -dir /Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/SDF/',aid,'/',aid,'_',i,'.sdf'))
  
}

######################### Create Dataset ##############################

aid = "AID_485314"  # change AID

print(paste0("Step 1: Started to analyze ", aid))

path = paste0("/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/CSV/",aid,"_datatable_all.csv")
comps = read.csv(path, header = T, stringsAsFactors = F)
dim(comps)
table(comps$PUBCHEM_ACTIVITY_OUTCOME)

comps2 = comps[,1:4]
head(comps2)
dim(comps2)

comps2 = comps2[complete.cases(comps2),]
dim(comps2)

table(comps2$PUBCHEM_ACTIVITY_OUTCOME)
head(comps2)

comps3 = comps2[,c("PUBCHEM_CID","PUBCHEM_ACTIVITY_OUTCOME")]
head(comps3)
dim(comps3)

table(comps3$PUBCHEM_ACTIVITY_OUTCOME)

dups = duplicated(comps3$PUBCHEM_CID)
comps4 = comps3[!dups,]
comps4 = comps4[complete.cases(comps4),]
head(comps4)
dim(comps4)
rownames(comps4) = comps4$PUBCHEM_CID
table(comps4$PUBCHEM_ACTIVITY_OUTCOME)

cids = unique(as.numeric(comps4$PUBCHEM_CID))

library(dplyr)
library(data.table)

data = read.csv(paste0("~/Documents/Studies/DeepDrug/PaDEL-Descriptor/Result/",aid,"/",aid,"_1.csv"), header = T)

f = list.files(paste0("~/Documents/Studies/DeepDrug/PaDEL-Descriptor/Result/",aid,"/"))

for(i in 2:length(f)){
  
  data2 = read.csv(paste0("~/Documents/Studies/DeepDrug/PaDEL-Descriptor/Result/",aid,"/",aid,"_",i,'.csv'), header = T)
  data = data
  if(nrow(data2)>0){
    data <- data.table::rbindlist(list(data,data2))
  }
  print(i)
}

dim(data)
path = paste0("~/Documents/Studies/DeepDrug/PaDEL-Descriptor/dataset/",aid,"/beforePreprocess/",aid,".txt")
fwrite(data, path, sep = "\t")


dim(comps4)
comps4[1:4,1:2]


comps5 = comps4[(rownames(comps4) %in% data$Name),]
dim(comps5)
colnames(comps5)[1] = "Name"
head(comps5)

comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Active"] <- "1"
comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Inactive"] <- "0"
comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Inconclusive"] <- NA
table(comps5$PUBCHEM_ACTIVITY_OUTCOME)

table(rownames(comps5) %in% data$Name)

dataset = merge(data, comps5, by = "Name")
dim(dataset)
dataset$PUBCHEM_ACTIVITY_OUTCOME = as.factor(dataset$PUBCHEM_ACTIVITY_OUTCOME)
table(dataset$PUBCHEM_ACTIVITY_OUTCOME)

dataset2 = dataset[,-1]

dataset2 = dataset2[!is.na(dataset2$PUBCHEM_ACTIVITY_OUTCOME),]
dim(dataset2)


dataset3 = as.data.frame(dataset2)
class(dataset3)

vNames = names(dataset3)[colSums(is.na(dataset3)) != nrow(dataset3)]
head(vNames)

dataset4 = dataset3[,vNames]
dim(dataset4)
dataset4[1:4,1:4]


splitData = split(dataset4, dataset4$PUBCHEM_ACTIVITY_OUTCOME)

inactives = as.data.frame(splitData$`0`)
actives = as.data.frame(splitData$`1`)


################### Preprocess Actives #############
actives2 <- data.frame(sapply( actives, as.numeric ))

vNames = names(actives2)[colSums(is.na(actives2)) != nrow(actives2)]
length(vNames)

actives3 = actives2[,vNames]
dim(actives3)

preProcActives = caret::preProcess(actives3, method = "nzv")

actives4 = actives3[,!(colnames(actives3) %in% preProcActives$method$remove)]
dim(actives4)
actives4[1:4,1:4]

actives4$class = rep(1,nrow(actives4))

if(anyNA(actives4)){
  
  for(i in 1:ncol(actives4)){
    actives4[is.na(actives4[,i]), i] <- median(actives4[,i], na.rm = TRUE)
    print(i)
  }
  
}


################### Preprocess Inactives #############
inactives2 <- data.frame(sapply( inactives, as.numeric ))

vNames = names(inactives2)[colSums(is.na(inactives2)) != nrow(inactives2)]
length(vNames)

inactives3 = inactives2[,vNames]
dim(inactives3)

preProcinactives = caret::preProcess(inactives3, method = "nzv")

inactives4 = inactives3[,!(colnames(inactives3) %in% preProcinactives$method$remove)]
dim(inactives4)
inactives4[1:4,1:4]

inactives4$class = rep(0,nrow(inactives4))
anyNA(inactives4)

if(anyNA(inactives4)){
  
  for(i in 1:ncol(inactives4)){
    inactives4[is.na(inactives4[,i]), i] <- median(inactives4[,i], na.rm = TRUE)
    print(i)
  }
  
}


##################  Merge actives and inactives ###################

inactives5 = inactives4[, colnames(inactives4) %in% colnames(actives4)]
dim(inactives5)

actives5 = actives4[, colnames(actives4) %in% colnames(inactives4)]
dim(actives5)

dataset5 = data.frame(data.table::rbindlist(list(inactives5, actives5)))
dim(dataset5)
table(dataset5$class)

dataset6 = dataset5[complete.cases(dataset5), ]
dim(dataset6)

sapply(dataset5, is.infinite)


for(i in 1:nrow(dataset5)){
  
  if(table(is.finite(dataset5[,i]))[[1]] != nrow(dataset5)){
    
    print(names(dataset5[i]))
    
  }
  
}

for (j in 1:ncol(dataset5)) set(dataset5, which(is.infinite(dataset5[[j]])), j, NA)

dataset6 = dataset5[complete.cases(dataset5),]
dim(dataset5)
dim(dataset6)

table(dataset6$class)

dim(dataset6) == dim(dataset6[complete.cases(dataset6),])


write.table(dataset6, paste0("/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/dataset/",aid,"/",aid,"_data.txt"), quote = F, row.names = F, sep = "\t")

data.table::fread(paste0("/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/dataset/",aid,"/",aid,"_data.txt"))

