rm(list = ls())

################### Get dataset #############

aid = "AID_485314" #change AID

path = paste0("/Users/selcukkorkmaz/Documents/Studies/DeepDrug/PaDEL-Descriptor/CSV/",aid,
              "_datatable_all.csv")
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


library(dplyr)
library(data.table)

path = paste0("/Volumes/selcuk/DeepDrug/dataset/",aid,"/beforePreprocess/",aid,".txt")

data = fread(path,sep = "\t")
dim(data)

dim(comps4)
comps4[1:4,1:2]

comps5 = comps4[(rownames(comps4) %in% data$Name),]
dim(comps5)
colnames(comps5)[1] = "Name"
head(comps5)

comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Active"] <- "1"
comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Inactive"] <- "0"
comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Inconclusive"] <- NA
comps5$PUBCHEM_ACTIVITY_OUTCOME[comps5$PUBCHEM_ACTIVITY_OUTCOME == "Unspecified"] <- NA

table(comps5$PUBCHEM_ACTIVITY_OUTCOME)

table(rownames(comps5) %in% data$Name)

dataset = merge(data, comps5, by = "Name")
dim(dataset)
# dataset$PUBCHEM_ACTIVITY_OUTCOME = as.factor(dataset$PUBCHEM_ACTIVITY_OUTCOME)
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

dim(inactives)
dim(actives)

### Remove NA columns from actives #####

# replace INFs with NAs
for (j in 1:ncol(actives)) set(actives, which(is.infinite(actives[[j]])), j, NA)

# remove all NA columns
vNames = names(actives)[colSums(is.na(actives)) != nrow(actives)]
length(vNames)
actives2 = actives[,vNames]
dim(actives2)


### Remove NA columns from inactives #####

# replace INFs with NAs
for (j in 1:ncol(inactives)) set(inactives, which(is.infinite(inactives[[j]])), j, NA)

# remove all NA columns
vNames = names(inactives)[colSums(is.na(inactives)) != nrow(inactives)]
length(vNames)
inactives2 = inactives[,vNames]
dim(inactives2)


##################  Merge ###################

inactives3 = inactives2[, colnames(inactives2) %in% colnames(actives2)]
dim(inactives3)

actives3 = actives2[, colnames(actives2) %in% colnames(inactives2)]
dim(actives3)

dataset5 = data.frame(data.table::rbindlist(list(inactives3, actives3)))
dim(dataset5)
table(dataset5$PUBCHEM_ACTIVITY_OUTCOME)

datasetM = dataset5[complete.cases(dataset5),]
dim(datasetM)
table(datasetM$PUBCHEM_ACTIVITY_OUTCOME)



#### Remove zero variance variables ####
zv = list()

for(i in 1:ncol(datasetM)){
  
  sdev = sd(datasetM[,i])
  
  if(sdev == 0){
    
    zv[[i]] = i
  }
  
  print(i)
}

zvariances = unlist(zv, use.names=FALSE)
length(zvariances)

dataset6 = datasetM[,-zvariances]
dim(dataset6)

dataset6[1:4,1:10]
table(dataset6$PUBCHEM_ACTIVITY_OUTCOME)


# ### Z-score transformation ####
# 
# # Find numerical variables for z-score transformation
# 
zList = list()
 
for(i in 1:ncol(dataset6)){

  if(length(unique(dataset6[,i])) > 2){

    zList[[i]] = colnames(dataset6[i])


  }

  print(i)

}

zTransform = unlist(zList)
length(zTransform)

# apply z-score transformation

for(i in 1:(ncol(dataset6))){
  
  if(colnames(dataset6[i]) %in% zTransform){

  dataset6[,i] = scale(dataset6[,i])

  }

  print(i)

}

dataset6[1:4,1:10]
dataset6[1:4,1940:1943]

dim(dataset6)

#### Write the dataset to a file ####

X = dataset6[,1:(ncol(dataset6)-1)]
Y = dataset6[ncol(dataset6)]

path = paste0("/Volumes/selcuk/DeepDrug/dataset/",aid,"/afterPreprocess/",aid,".txt")
pathX = paste0("/Volumes/selcuk/DeepDrug/dataset/",aid,"/afterPreprocess/X.txt")
pathY = paste0("/Volumes/selcuk/DeepDrug/dataset/",aid,"/afterPreprocess/Y.txt")

fwrite(dataset6, path, sep = "\t")
fwrite(X, pathX, sep = "\t")
fwrite(Y, pathY, sep = "\t")





