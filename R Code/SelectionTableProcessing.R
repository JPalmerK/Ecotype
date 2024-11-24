library(pracma)
library(ggplot2)
precisionRecall<- function(selTable, 
                           truthTable, 
                           class = 'KW_SRKW', 
                           mergeMethod = NA,
                           scoreSteps = .01){
  
  # Create vectors to store precision and recall
  precisions <- c()
  recalls <- c()
  
  selTable = selTable[selTable$`Species Code`== class,]
  
  # Confidence scores
  conScores = seq(min(selTable$Confidence, na.rm = TRUE), 
                  max(selTable$Confidence),by= scoreSteps)
  
  for (conScore in conScores) {
    
    # Filter selection table based on confidence score and class. 
    selTableSub <- subset(selTable, 
                          Confidence >= conScore)
    
    # Merge the selections
    if(mergeMethod %in% c("loglik", 'median')){
    merged_selections <- merge_selections(selTableSub, 
                                          score_method =mergeMethod)}else{
    merged_selections<-selTableSub}
    
    # Initialize counts for true positives, false positives, and false negatives
    tp <- 0
    fp <- nrow(merged_selections)
    fn <- nrow(truthTable)
    
    # Loop through each truth detection
    for (i in 1:nrow(truthTable)) {
      truth_start <- truthTable$`Begin Time (s)`[i]
      truth_end <- truthTable$`End Time (s)`[i]
      
      # Check if the center time of the truth detection falls within any merged selection
      for (j in 1:nrow(merged_selections)) {
        model_start <- merged_selections$`Begin Time (s)`[j]
        model_end <- merged_selections$`End Time (s)`[j]
        
        if (truth_start >= model_start && truth_start <= model_end) {
          tp <- tp + 1  # Found a true positive
          fn <- fn - 1  # Decrease false negatives count
          fp <- fp - 1  # Decrease false positives count
          break  # Stop checking further if we found a match
        }
      }
    }
    
    # Calculate precision and recall
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    
    # Store the results
    precisions <- c(precisions, precision)
    recalls <- c(recalls, recall)
    
    print(conScore)
  }
  
  # Create a data frame for the precision-recall results
  pr_results <- data.frame(Recall = recalls, 
                           Precision = precisions, 
                           Confidence = conScores,
                           MergeMethod = mergeMethod)
  return(pr_results)
}





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
    overlapIdx =which(selTable$`End Time (s)`+1.5>=truthTable$CenterTime[ii] &
                        selTable$`Begin Time (s)`-1.5<= truthTable$CenterTime[ii])
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



merge_scores <- function(score_vector, method = "median") {
  if (length(score_vector) == 0) {
    return(NA)  # Return NA if no scores are provided
  }
  
  if (method == "median") {
    return(median(score_vector))
  } else if (method == "loglik") {
    # Handle log likelihood merging
    log_scores <- log(score_vector)  # Convert scores to log space
    average_log_likelihood <- mean(log_scores)  # Calculate average log likelihood
    return(exp(average_log_likelihood))  # Convert back to probability scale
  } else {
    stop("Invalid method. Choose either 'median' or 'loglik'.")
  }
}

merge_selections <- function(selection_table, score_method = "median") {
  # Ensure the selection table is sorted by begin time
  selection_table <- selection_table[order(selection_table$`Begin Time (s)`), ]
  
  # Create an empty list to store the merged selections
  merged_selections <- list()
  
  # Initialize the first selection
  current_selection <- selection_table[1, ]
  
  # Create a vector to collect scores
  score_vector <- current_selection$Confidence  # Initialize with the first score
  
  # Loop through each selection
  for (i in 2:nrow(selection_table)) {
    next_selection <- selection_table[i, ]
    
    # Calculate the end time of the current selection
    current_end_time <- current_selection$`End Time (s)`
    
    # Check if the next selection overlaps or is adjacent (within 3 seconds)
    if (next_selection$`Begin Time (s)` <= current_end_time) {
      # If overlapping, extend the current selection and combine the scores
      current_selection$end_time <- max(current_selection$`End Time (s)`,
                                        next_selection$`End Time (s)`)
      # Collect scores
      score_vector <- c(score_vector, next_selection$Confidence)
      
    } else {
      # If no overlap, finalize the current selection
      current_selection$Confidence <- merge_scores(score_vector, 
                                                   method = score_method)  # Merge scores using the specified method
      current_selection$end_time <- current_selection$`End Time (s)`
      
      # Save the merged selection
      merged_selections[[length(merged_selections) + 1]] <- current_selection  
      
      # Move to the next selection and reset the score vector
      current_selection <- next_selection
      score_vector <- current_selection$Confidence  # Initialize with the new selection's score
    }
  }
  
  # Finalize the last selection
  current_selection$Confidence <- merge_scores(score_vector, method = score_method)  # Merge the last selection's scores
  current_selection$end_time <- current_selection$`End Time (s)`
  merged_selections[[length(merged_selections) + 1]] <- current_selection
  
  # Convert the list of merged selections back into a data frame
  merged_table <- do.call(rbind, merged_selections)
  
  return(merged_table)
}

merge_selections_by_class <- function(selection_table,  score_method = "median") {
  # Ensure the selection table has a column for class/species
  if (!"Species Code" %in% colnames(selection_table)) {
    stop("The selection table must have a 'Species Code' column")
  }
  
  # Split the selection table by species/class
  class_groups <- split(selection_table, selection_table$`Species Code`)
  
  # Create an empty list to store merged selections for all classes
  merged_tables <- list()
  
  # Apply the merge function to each class group
  for (class_name in names(class_groups)) {
    class_table <- class_groups[[class_name]]
    
    if (nrow(class_table) > 1) {
      # Apply the merge_selections function to each class
      merged_class_table <- merge_selections(class_table,  score_method)
    } else {
      # Ensure a single row has the necessary columns
      merged_class_table <- class_table
      merged_class_table$end_time <- merged_class_table$`End Time (s)`  # Add end_time if missing
      merged_class_table$Confidence <- merged_class_table$Confidence  # Ensure Confidence exists
    }
    
    # Add the species/class name back to the merged selections
    merged_class_table$species <- class_name
    
    # Store the merged class table
    merged_tables[[length(merged_tables) + 1]] <- merged_class_table
  }
  
  # Combine all the merged class tables into a single data frame
  final_selection_table <- do.call(rbind, merged_tables)
  
  return(final_selection_table)
}

merge_peaks <- function(df, threshold = 1) {
  
  # This function merges peaks derived from the peakfinder PRACMA library, 
  # data must be in a data frame with x1 and x2 values and merges adjacent 
  # detections
  
  # Sort the dataframe by the start index (x1)
  df <- df[order(df$x1),]
  
  # Initialize a list to hold merged peaks
  merged_peaks <- list()
  
  # Start with the first peak
  current_peak <- df[1, ]
  
  for (i in 2:nrow(df)) {
    next_peak <- df[i, ]
    
    # Check if the difference between A$x2 and B$x1 is less than the threshold
    if (abs(current_peak$x2 - next_peak$x1) < threshold) {
      # Merge the peaks by taking min(x1) and max(x2)
      current_peak$x1 <- min(current_peak$x1, next_peak$x1)
      current_peak$x2 <- max(current_peak$x2, next_peak$x2)
    } else {
      # If they don't meet the threshold, save the current peak and move to the next one
      merged_peaks <- rbind(merged_peaks, current_peak)
      current_peak <- next_peak
    }
  }
  
  # Add the last peak
  merged_peaks <- rbind(merged_peaks, current_peak)
  
  # Convert to dataframe
  merged_peaks_df <- as.data.frame(merged_peaks)
  return(merged_peaks_df)
}


pointPicking<-function(aa, tstep =0.25, thresh = 0.8){
  'Function to pick points out of birdnet output. AA is selection table with
  `Begin Time (s)` and Confidence values produced by Birdnet'
  

  # Ok well this isn't quite working because we have a detection threshold of 5
  # so what we really need is a complete time series
   aa$BeginTimeRounded <- round_to_nearest(aa$`Begin Time (s)`,tstep)
  #Time series 
  tsDataframe = data.frame(tt = seq(0, max(aa$`End Time (s)`), 
                                    by =0.25), Confidence = 0.5)
  
  # Populate the dataframe
  for(ii in 1:nrow(aa)){
    tsDataframe$Confidence[tsDataframe$tt== aa$BeginTimeRounded [ii]]<-
      aa$Confidence[ii]
  }
  tsDataframe$ConfidenceMode = 10000^tsDataframe$Confidence
  
  
  signal =tsDataframe$ConfidenceMode
  peakVals <- as.data.frame(
    findpeaks(signal, 
              minpeakheight=10000^(thresh), 
              sortstr=TRUE,
              minpeakdistance=1))
  
  # peakVals <- as.data.frame(
  #   findpeaks(signal,threshold =10000^(thresh),
  #             sortstr=TRUE))
  colnames(peakVals)<-c('y', 'x', 'x1', 'x2')
  peakVals$tt = peakVals$x/4-0.25
  peakVals$dur = (peakVals$x2- peakVals$x1)*0.25
  max(peakVals$dur)
  
  # Convert to selection table format
  # Cool, lets see if we can turn this into a raven selection table
  peakVals$Selection =1:nrow(peakVals)
  peakVals$View = 'Spectrogram'
  peakVals$Channel = 1
  peakVals$`Begin Time (s)` = (peakVals$x1)/4-0.25
  peakVals$`End Time (s)` = (peakVals$x2)/4-0.25
  peakVals$`Low Freq (Hz)` = 0
  peakVals$`High Freq (Hz)` = 8000
  peakVals$Confidence = log(peakVals$y, base=10000)
  peakVals$Confidence = peakVals$y
  peakVals$`Species Code` = "KW_BKW"
  
  
  return(peakVals)
  
}

# First round the output values to the nearest 0.25
round_to_nearest <- function(x, base) {
  round(x / base) * base
}


##############################################################################
# Create merged selection tables
##############################################################################


# Read in the table
selTableFiltered = 'C:\\Users\\kaity\\Desktop\\BirdnetTest 20240924\\BirdNET_SelectionTable.txt'
selTableFiltered = read.table(selTableFiltered, header = TRUE, sep = '\t', check.names=FALSE)


selTableAllData = 'C:\\Users\\kaity\\Desktop\\BirdnetTest 20240924\\allData/BirdNET_SelectionTable.txt'
selTableAllData = read.table(selTableAllData, header = TRUE, sep = '\t', check.names=FALSE)
selTableAllData$`Species Code`[selTableAllData$`Species Code` =="Orcinus orcaSRKW_Killer whale SRKW" ] = 'KW_SRKW'
selTableAllData$`Species Code`[selTableAllData$`Species Code` =="Orcinus orcaBiggs_Killer whale Biggs" ] = 'KW_BKW'


################################################################################
# Read in the truth table and calculate Precision Recall
###############################################################################


# Read in the truth table
truthTable_BKW = 'C:\\Users\\kaity\\Desktop\\BirdnetTest 20240924\\MALAHAT_STN3_20160706T085506Z_SRKW_truth.txt'
truthTable_BKW = read.table(truthTable_BKW, header = TRUE, sep = '\t', check.names=FALSE)



# Read in the truth table
truthTable_SRKW = 'C:\\Users\\kaity\\Desktop\\BirdnetTest 20240924\\MALAHAT_STN3_20160709_SRKW_truth.txt'
truthTable_SRKW = read.table(truthTable_SRKW, header = TRUE, sep = '\t', check.names=FALSE)


# 
# #####################################################################
# # Create a precision recall curve. 
# #####################################################################
# #library(PRROC)  # Load the PRROC library
# library(ggplot2)
# methods = c('None',  'loglik')
# 
# pr_results= data.frame()
# pr_resultsAll = data.frame()
# 
# # Create the precision recall curves for each set of training data
# # Malahat Station 3, 2016-07-06 all transients 
# 
# for(mergeMethod in methods){
#   pr_results = rbind(pr_results,  precisionRecall(selTable, truthTable, 
#                                        class = 'KW_BKW', 
#                                        mergeMethod = mergeMethod))
#   
#   pr_resultsAll = rbind(pr_resultsAll,  precisionRecall(selTableAllData, truthTable, 
#                                                   class = 'KW_BKW', 
#                                                   mergeMethod = mergeMethod))
# }
# 
# 
# 
# 
# pr_results$Model = 'CleanedData'
# pr_resultsAll$Model = 'AllData'
# 
# allData = rbind(pr_results,pr_resultsAll)
# 
# PrPLOT<-ggplot(allData)+geom_point(aes(y = Precision, x= Recall, 
#                                color =  MergeMethod))+
#   facet_grid(~Model)+
#   ggtitle('Birdnet Experiment 1 day Biggs Recordings Malahat')
# 
# PrPLOT
##########################################################################
# Plot the detection scores
# ###########################################################################
# aa = selTableAllData[selTableAllData$`Species Code`== 'KW_BKW',]
# bb = selTableAllData[selTableAllData$`Species Code`== 'KW_SRKW',] 
# 
# ggplot()+
#   geom_point(data = aa, aes(x= `Begin Time (s)`, 
#                             y=(Confidence)), color= 'blue')+
#   geom_point(data = bb, aes(x= `Begin Time (s)`, 
#                             y=(Confidence)), color = 'red')+
#   geom_point(data = truthTable, aes(x= `Begin Time (s)`, 
#                                     y= (1)), color = 'black')+
#   xlim(600,900)
# 
# # Looks like scaling the values this way may be a valuable transformation
# ggplot()+
#   geom_point(data = aa, aes(x= `Begin Time (s)`, 
#                             y=1000^(Confidence)), color= 'blue')+
#   geom_point(data = bb, aes(x= `Begin Time (s)`, 
#                             y=1000^(Confidence)), color = 'red')+
#   geom_point(data = truthTable, aes(x= `Begin Time (s)`, y= 1000^(1)), color = 'black')+
#   xlim(600,900)





###################################################################
# Try the existing peak picking algorithim, Bigs
##########################################################################

BKW = selTableFiltered[selTableFiltered$`Species Code`== 'KW_BKW',]
SRKW = selTableFiltered[selTableFiltered$`Species Code`== 'KW_SRKW',]

peakVals_BKW<-pointPicking(BKW, tstep =0.25)
peakVals_SRKW<-pointPicking(SRKW, tstep =0.25)


peakVals_BKW<-merge_peaks(peakVals_BKW, threshold = 1)
peakVals_SRKW<-merge_peaks(peakVals_SRKW, threshold = 1)

# fileName = 'PeakPickerTestOutputFilteredDataMerged.txt'
# write.table(peakVals1, sep = '\t',
#             file = fileName, row.names = FALSE,
#             col.names = TRUE, quote = FALSE)



#############################################################################
# Create Precision Recall for peak picked calls
#############################################################################

precisionRecall_SRKW <- precisionRecall_MatrixMethod(peakVals_SRKW,
                                                      truthTable = truthTable_BKW)
precisionRecall_SRKW$ModelOut = 'SRKW'

precisionRecall_BKW <- precisionRecall_MatrixMethod(peakVals_BKW,
                                                     truthTable = truthTable_BKW)
precisionRecall_BKW$ModelOut = 'BKW'

allPrecisionRecall = rbind(precisionRecall_SRKW,precisionRecall_BKW)


ggplot(allPrecisionRecall, aes(x =Recall, y = Precision, color= ModelOut))+
  geom_point()+
  ggtitle('Birdnet Filtered Data BKW Eval')+
  xlim(0,1)+ylim(0,1)
  

###########################################################################
# Try the birdnet data from the unfiltered data
###########################################################################
BKW_all = selTableAllData[selTableAllData$`Species Code`== 'KW_BKW',]
SRKW_all = selTableAllData[selTableAllData$`Species Code`== 'KW_SRKW',]

peakVals_BKW<-pointPicking(BKW_all, tstep =0.25)
peakVals_SRKW<-pointPicking(SRKW_all, tstep =0.25)


peakVals_BKW_all<-merge_peaks(peakVals_BKW, threshold = 1)
peakVals_SRKW_all<-merge_peaks(peakVals_SRKW, threshold = 1)

# fileName = 'PeakPickerTestOutputAllDataMerged.txt'
# write.table(peakVals1, sep = '\t',
#             file = fileName, row.names = FALSE,
#             col.names = TRUE, quote = FALSE)


precisionRecall_SRKW <- precisionRecall_MatrixMethod(peakVals_SRKW_all,
                                                     truthTable = truthTable_BKW)
precisionRecall_SRKW$ModelOut = 'SRKW'

precisionRecall_BKW <- precisionRecall_MatrixMethod(peakVals_BKW_all,
                                                    truthTable = truthTable_BKW)
precisionRecall_BKW$ModelOut = 'BKW'

allPrecisionRecall = rbind(precisionRecall_SRKW,precisionRecall_BKW)


ggplot(allPrecisionRecall, aes(x =Recall, y = Precision, color= ModelOut))+
  geom_point()+
  ggtitle('Birdnet All Training Data BKW Eval')+
  xlim(0,1)+ylim(0,1)











# 
# precisionRecall(peakVals, truthTable, 
#                 class = 'KW_BKW', 
#                 mergeMethod = mergeMethod)
# 
# 
# 
# methods = c('None',  'loglik')
# 
# pr_results= data.frame()
# pr_resultsAll = data.frame()
# 
# # Create the precision recall curves for each set of training data
# # Malahat Station 3, 2016-07-06 all transients 
# 
# for(mergeMethod in methods){
#   pr_results = rbind(pr_results,  precisionRecall(selTable, truthTable, 
#                                                   class = 'KW_BKW', 
#                                                   mergeMethod = mergeMethod))
#   
#   pr_resultsAll = rbind(pr_resultsAll,  precisionRecall(selTableAllData, truthTable, 
#                                                         class = 'KW_BKW', 
#                                                         mergeMethod = mergeMethod))
# }
# 
# pr_results$Model = 'CleanedData'
# pr_resultsAll$Model = 'AllData'
# 
# allData = rbind(pr_results,pr_resultsAll)
# 
# ggplot(allData)+geom_point(aes(y = Precision, x= Recall, 
#                                color =  MergeMethod))+
#   facet_grid(~Model)+
#   ggtitle('Birdnet Experiment 1 day Biggs Recordings Malahat')
# 
# 
# 
