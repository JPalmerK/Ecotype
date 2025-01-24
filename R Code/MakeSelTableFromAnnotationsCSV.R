library(dplyr)
library(lubridate)
library(ggplot2)
library(av)
# Get a list of files matching the pattern 'annot_Malahat'
file_list <- list.files(path = 'E:/DCLDE/Malahat_JASCO/Annotations/',
                        pattern = 'annot_Malahat', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
JASCO_malahat <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.csv(file)
  if (nrow(data) > 0) {
    data$Dep <- as.factor(basename(file))  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))


JASCO_malahat <- JASCO_malahat %>%
  mutate(
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename),
                     format = "%Y%m%dT%H%M%S", tz= 'UTC'))

JASCO_malahat$UTC = JASCO_malahat$UTC+
  seconds(as.numeric(JASCO_malahat$start))

JASCO_malahat$ClassSpecies = as.factor(JASCO_malahat$sound_id_species)
levels(JASCO_malahat$ClassSpecies)<-c('AB', 'HW','KW','HW',  'KW', 'KW', 
                                      'UndBio', 'UndBio','AB', 'AB', 'AB')

JASCO_malahat$Ecotype = as.factor(JASCO_malahat$kw_ecotype)
levels(JASCO_malahat$Ecotype)<- c(NA,'SRKW', NA, 'TKW', NA)

JASCO_malahat$KW =ifelse(JASCO_malahat$ClassSpecies %in% c('KW', 'KW?'), 1, 0)

JASCO_malahat$KW_certain = NA
JASCO_malahat$KW_certain[JASCO_malahat$KW ==1] =1
JASCO_malahat$KW_certain[JASCO_malahat$sound_id_species %in% c("HW/KW?",
                                                               "KW?")]=0

colnames(JASCO_malahat)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                         'FileEndSec', 'Soundfile')

JASCO_malahat$AnnotationLevel = 'Call'

JASCO_malahat$dur = JASCO_malahat$FileEndSec-JASCO_malahat$FileBeginSec

JASCO_malahat$end_time = JASCO_malahat$UTC+ seconds(JASCO_malahat$dur)




JASCO_malahat$Provider = 'JASCO_Malahat'
levels(JASCO_malahat$Dep)<-c('STN3', 'STN4', 'STN5', 'STN6')

# Filepaths
dayFolderPath = 'E:\\Malahat'
JASCO_malahat$FilePath =
  file.path('E:/Malahat', JASCO_malahat$Dep,JASCO_malahat$Soundfile)

JASCO_malahat$FileOk  = file.exists(JASCO_malahat$FilePath)


# Make sure all audio files are present for all annotations
if (all(JASCO_malahat$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


# Nix annotations without audio files
""
JASCO_malahat= JASCO_malahat[JASCO_malahat$FileOk==TRUE,]

############################################################################
# Create a raven selection table for each deployment
SeltableData = JASCO_malahat[JASCO_malahat$Dep== 'STN3',]


# List of audio files
 audio.files = data.frame(
   filename = list.files('E:\\Malahat\\STN3',
            pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
 audio.files$Soundfile =basename(audio.files$filename)
 audio.files$filepath = file.path('E:\\Malahat\\STN3', audio.files$filename)
 
 audio.files$Index =1:nrow(audio.files)
 audio.files= audio.files[,c(2,3)]
 
 
 audio.files$Duration = as.numeric(sapply(audio.files$filepath, 
                                        av_media_info)["duration", ]) 
 
 audio.files$FileStart =0
 for(ii in 2:nrow(audio.files)){
   audio.files$FileStart[ii]<- audio.files$FileStart[ii-1]+
     audio.files$Duration[ii-1]
 }
 
 SeltableData = merge(SeltableData, audio.files, by.x = 'Soundfile',
            by.y= 'Soundfile')
 
 SeltableData$'Begin Time (s)' = SeltableData$FileStart+ SeltableData$FileBeginSec
 SeltableData$'End Time (s)' = SeltableData$FileStart+ SeltableData$FileEndSec
 SeltableData$'Low Freq (Hz)' = SeltableData$LowFreqHz
 SeltableData$'High Freq (Hz)' = SeltableData$HighFreqHz
 
 SeltableData$Selection = 1:nrow(SeltableData)
 SeltableData$Channel =1
 SeltableData$View =1

 
 SelOut = SeltableData[,c(31,32,33,27,28,29,30,8,9)]
 
 write.table(SelOut,
             "C:/TempData/DCLDE_EVAL/Malahat_JASCO/Annotations/Station_3_updated.txt",
             sep="\t",row.names=FALSE, quote = FALSE)









