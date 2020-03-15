###############################################
aid = "AID_485341"
method = "rus"

path = paste0("/Volumes/selcuk/DeepDrug/Classification Results/",aid,"/",method,"_val.txt")

data = read.table(path, header = T)
head(data)
dim(data)

library("ROCit")


measure <- measureit(score = data$Prob, class = data$True,
                     measure = c("SENS", "SPEC", "FSCR"))

measure$BACC = ((measure$TP/(measure$TP+measure$FN))+(measure$TN/(measure$TN+measure$FP)))/2


meas = cbind.data.frame(cutoff = measure$Cutoff, score = measure$BACC)

cutoff = meas[which.max(meas$score),][["cutoff"]]


plot(measure$Cutoff, measure$SPEC, type = "l")
lines(measure$Cutoff,measure$SENS,col="green")
abline(v=cutoff, col = "red")

plot(measure$Cutoff, measure$BACC, type = "l", xlab = "Threshold", 
     ylab = "Balanced Accuracy", main = paste0("Optimal threshold for ", aid))
abline(v=cutoff, col = "red")


path = paste0("/Volumes/selcuk/DeepDrug/Classification Results/",aid,"/",method,"_test.txt")

data = read.table(path, header = T)
head(data)
dim(data)


data$Pred = ifelse(data$Prob > cutoff, 1,0)

perfs = caret::confusionMatrix(as.factor(data$Pred), as.factor(data$True), positive = '1')


table = perfs$table
tn = table[1,1]
tp = table[2,2]
fp = table[2,1]
fn = table[1,2]

bacc = perfs$byClass[["Balanced Accuracy"]]
precision = perfs$byClass[["Pos Pred Value"]]
recall = perfs$byClass[["Sensitivity"]]
f1 = 2*precision*recall/(precision+recall)
mcc = as.numeric(tp*tn-fp*fn) / 
  (sqrt(as.numeric(tp+fp)*as.numeric(tp+fn)*as.numeric(tn+fp)*as.numeric(tn+fn)) )

result = cbind.data.frame(cutoff = cutoff, bacc,precision,recall,f1,mcc)

path2 = paste0("/Volumes/selcuk/DeepDrug/Performance/",aid,"/metrics/",method,"_metrics.txt")
path3 = paste0("/Volumes/selcuk/DeepDrug/Performance/",aid,"/table/",method,"_table.txt")

write.table(result, path2, quote = F, row.names = F, sep = "\t")
write.table(table, path3, quote = F, row.names = F, sep = "\t")
  
result
table


