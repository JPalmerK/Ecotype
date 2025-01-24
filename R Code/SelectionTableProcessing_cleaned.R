
library(ggplot2)
library(patchwork) # To display 2 charts together

precisionRecall_MatrixMethod <- function(selTable, truthTable, 
                                         mergeMethod= NA,
                                         scoreLen =50){
  
  # Create precision recall by creating a matrix where rows are the observed
  # detections and columns are the truth. 
  
  truthTable$CenterTime = truthTable$`Begin Time (s)`+
    (truthTable$`End Time (s)`-truthTable$`Begin Time (s)`)/2
  
  DetectionMat = matrix(0,nrow = nrow(selTable), ncol = nrow(truthTable))
  
  
  # Now step through the truth matrix and see which ones overlap 
  for (ii in 1:ncol(DetectionMat)){
    overlapIdx =which(selTable$`End Time (S)`+1>=truthTable$CenterTime[ii] &
                        selTable$`Begin Time (S)`-1<= truthTable$CenterTime[ii])
    DetectionMat[overlapIdx,ii] = 1}
  
  
  # Confidence scores
  conScores = as.numeric(quantile(selTable$Confidence, 
                                  seq(0,.99, length.out =scoreLen)))
  
  precisionRecall = data.frame(
    scores = conScores[1:length(conScores)-1],
    tp = rep(0, length(conScores)-1),
    fp = rep(0, length(conScores)-1),
    fn = rep(0, length(conScores)-1),
    ndet = rep(0, length(conScores)-1)
  )
  
  for(ii in 1:length(conScores)-1){
    # All trugh columns only rows where the detection score is greater or equal
    # to the threshold
    detMatsub = DetectionMat[which(selTable$Confidence>=conScores[ii]),]
    
    # False negative, truth detections that were missed
    precisionRecall$fn[ii]= sum(colSums(detMatsub)==0)
    
    # False positives, model detections that did not mach any truth
    precisionRecall$fp[ii] =sum(rowSums(detMatsub)==0)
    
    # True positive, the number detections that matched a truth detection
    precisionRecall$tp[ii] = sum(detMatsub)
    
    # Total number of detections
    precisionRecall$ndet[ii] = nrow(detMatsub)
    print(ii)
    
  }
  precisionRecall$Precision =precisionRecall$tp/ (precisionRecall$tp + precisionRecall$fp) 
  precisionRecall$Recall =precisionRecall$tp/ (nrow(truthTable)) 
  
  return(precisionRecall)
}



# Read in the truth table
truthTable_SRKW = 'C:\\TempData/DCLDE_EVAL/SMRU/Annotations/LK_20210728_003000_000.Table.1.selections.txt'
truthTable_SRKW = read.table(truthTable_SRKW, header = TRUE, sep = '\t', check.names=FALSE)
# Keep only the SRKW calls from the truth table
truthTable_SRKW = truthTable_SRKW[truthTable_SRKW$kw_ecotype %in% c('SRKW', 'SRKW?'),]



# Read in the evaluation tables
selTableAllData = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\detections_20241126.txt'
selTableAllData = read.table(selTableAllData, header = TRUE, sep = '\t', check.names=FALSE)
selTableRKW = selTableAllData[selTableAllData$Class == 'RKW',]
selTableRKW$Confidence = selTableRKW$Score

precisionRecall_SRKW <- precisionRecall_MatrixMethod(selTableRKW,
                                                     truthTable = truthTable_SRKW)

precisionRecall_TKW<-precisionRecall_MatrixMethod(selTableRKW,
                                                  truthTable = truthTable_SRKW)


p1<-ggplot(precisionRecall_SRKW, aes(x =Recall, y = Precision))+
  geom_point()+
  ylim(0.25,0.6)+
  ggtitle('Resident KW Superpod Event')

# False postiives per hour
p2<-ggplot(precisionRecall_SRKW)+
  geom_point(aes(x = scores, y= precisionRecall_SRKW$fp/24))+
  xlabel(Scores)



p1+p2

##########################################################################
# Evaluate RKW and TKW for Malahat
###########################################################################


# Read in the truth table
truthTable = 'C:\\TempData/DCLDE_EVAL/Malahat_JASCO/Annotations/Station_3_valid.txt'
truthTable = read.table(truthTable, header = TRUE, 
                        sep = '\t', check.names=FALSE)
# Keep only the SRKW calls from the truth table
truthTable_SRKW = truthTable[truthTable$kw_ecotype %in% c('SRKW', 'SRKW?'),]
truthTable_TKW =truthTable[truthTable$kw_ecotype %in% c('TKW', 'TKW?'),]
  

# Read in the evaluation tables
selTableAllData = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\detections_Malahat_20241126.txt'
selTableAllData = read.table(selTableAllData, header = TRUE, sep = '\t', check.names=FALSE)
selTableAllData$Confidence = selTableAllData$Score
selTableAllData$`End Time (S)`= selTableAllData$`End Time (s)`
selTableAllData$`Begin Time (S)`= selTableAllData$`Begin Time (s)`

selTableRKW = selTableAllData[selTableAllData$Class == 'RKW',]
selTableTKW = selTableAllData[selTableAllData$Class == 'TKW',]

precisionRecall_SRKW <- precisionRecall_MatrixMethod(selTableRKW,
                                                     truthTable = truthTable_SRKW)
precisionRecall_SRKW$Class = 'SRKW'


precisionRecall_TKW<-precisionRecall_MatrixMethod(selTableTKW,
                                                  truthTable = truthTable_TKW)
precisionRecall_TKW$Class = 'TKW'

precisionRecall= rbind(precisionRecall_TKW, precisionRecall_SRKW)

p1<-ggplot(precisionRecall, aes(x =Recall, y = Precision, color = Class))+
  geom_point()+
  ggtitle('Malahat Station 3 Performance')

# False postiives per hour
p2<-ggplot(precisionRecall)+
  geom_point(aes(x = scores, y= fp/24, color = Class))+
  xlab('Scores')+
  ylab('False Positives per Hour')

p3<-ggplot(precisionRecall, aes(x =scores, y = Recall, color = Class))+
  geom_point()


library(patchwork) # To display 2 charts together

p1+p2

##############################################################################
# 8khz model trained with batch normalization 
##############################################################################

# Read in the truth table
truthTable = 'C:\\TempData/DCLDE_EVAL/Malahat_JASCO/Annotations/Station_3_valid.txt'
truthTable = read.table(truthTable, header = TRUE, 
                        sep = '\t', check.names=FALSE)
# Keep only the SRKW calls from the truth table
truthTable_SRKW = truthTable[truthTable$kw_ecotype %in% c('SRKW', 'SRKW?'),]
truthTable_TKW =truthTable[truthTable$kw_ecotype %in% c('TKW', 'TKW?'),]


# Read in the evaluation tables
selTableAllData = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\detections_Malahat_20241130.txt'
selTableAllData = read.table(selTableAllData, header = TRUE, sep = '\t', check.names=FALSE)
selTableAllData$Confidence = selTableAllData$Score
selTableAllData$`End Time (S)`= selTableAllData$`End Time (S)`
selTableAllData$`Begin Time (S)`= selTableAllData$`Begin Time (S)`

selTableRKW = selTableAllData[selTableAllData$Class == 'RKW',]
selTableTKW = selTableAllData[selTableAllData$Class == 'TKW',]

precisionRecall_SRKW <- precisionRecall_MatrixMethod(selTableRKW,
                                                     truthTable = truthTable_SRKW)
precisionRecall_SRKW$Class = 'SRKW'


precisionRecall_TKW<-precisionRecall_MatrixMethod(selTableTKW,
                                                  truthTable = truthTable_TKW)
precisionRecall_TKW$Class = 'TKW'

precisionRecall= rbind(precisionRecall_TKW, precisionRecall_SRKW)

p1<-ggplot(precisionRecall, aes(x =Recall, y = Precision, color = Class))+
  geom_point()+
  ggtitle('Malahat Station 3 Performance')

# False postiives per hour
p2<-ggplot(precisionRecall)+
  geom_point(aes(x = scores, y= fp/24, color = Class))+
  xlab('Scores')+
  ylab('False Positives per Hour')

p3<-ggplot(precisionRecall, aes(x =scores, y = Recall, color = Class))+
  geom_point()

