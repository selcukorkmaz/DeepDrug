aid = "AID_485341"

path_X_train_raw = paste0("/Volumes/selcuk/DeepDrug/dataset/resample/",aid,"/",aid,"/X_train_raw.txt")
path_Y_train_raw = paste0("/Volumes/selcuk/DeepDrug/dataset/resample/",aid,"/",aid,"/y_train_raw.txt")


data = data.table::fread("/Volumes/selcuk/DeepDrug/dataset/resample/AID_485314/AID_485314/X_train_raw.txt",
                  sep = ",", header = T)

dim(data)

class = data.table::fread("/Volumes/selcuk/DeepDrug/dataset/resample/AID_485314/AID_485314/Y_train_raw.txt",
                           header = T)



nrow(class) == nrow(data)

dataNew = dplyr::bind_cols(data,class)
dim(dataNew)

dataNew[1:4,1937:1938]
table(dataNew$PUBCHEM_ACTIVITY_OUTCOME)

indx = caret::createDataPartition(dataNew$PUBCHEM_ACTIVITY_OUTCOME, p = 0.10)

validation = dataNew[indx$Resample1,]
dim(validation)

X_validation_raw = validation[,-"PUBCHEM_ACTIVITY_OUTCOME"]
dim(X_validation_raw)
y_validation_raw = validation[,"PUBCHEM_ACTIVITY_OUTCOME"]
dim(y_validation_raw)

train = dataNew[-indx$Resample1,]
dim(train)
X_train_raw = train[,-"PUBCHEM_ACTIVITY_OUTCOME"]
dim(X_train_raw)
y_train_raw = train[,"PUBCHEM_ACTIVITY_OUTCOME"]
dim(y_train_raw)


path_X_val_raw = paste0("/Volumes/selcuk/DeepDrug/dataset/resample/",aid,"/",aid,"/X_val_raw.txt")
path_Y_val_raw = paste0("/Volumes/selcuk/DeepDrug/dataset/resample/",aid,"/",aid,"/y_val_raw.txt")


data.table::fwrite(X_train_raw, path_X_train_raw, sep = ",")
data.table::fwrite(y_train_raw, path_Y_train_raw, sep = ",")
data.table::fwrite(X_validation_raw, path_X_val_raw, sep = ",")
data.table::fwrite(y_validation_raw, path_Y_val_raw, sep = ",")

