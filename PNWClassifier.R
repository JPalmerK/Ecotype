#Process Annotations, PNW only, and certain Ecotype Only
rm(list =ls())
library(lubridate)
library(dplyr)

source('C:/Users/kaity/Documents/GitHub/DCLDE2026/TestFx.R')
# Data to pretty
# 
# 1) Ocean Networks Canada 
# 2) Viers
# 3) DFO Cetacean Research Program (Pilkington)
# 4) DFO Whale Detection and Localization (Yerk)
# 5) SMRU
# 6) VPFA
# 7) Scripps

############################################################################
# Final output column names

colOut = c('Soundfile','Dep','LowFreqHz','HighFreqHz','FileEndSec', 'UTC',
           'FileBeginSec','ClassSpecies','KW','KW_certain','Ecotype', 'Provider',
           'AnnotationLevel', 'FilePath', 'FileOk')

ClassSpeciesList = c('KW', 'HW', 'AB', 'UndBio')
AnnotationLevelList = c('File', 'Detection', 'Call')
EcotypeList = c('SRKW', 'BKW', 'OKW', 'NRKW')
# For the class species, options are: KW, HW, AB, and UndBio
# Killer whale, humpback whale, abiotic, and unknown Bio which includes 
# Other dolphin acoustically active species ranging from fin whales, to 
# white beaked dolphins to seagulls

###############################################################################
# 1) ONC- data from globus


# Jasper, April, and Jenn have all annotated thse files. There is some overla
#ONC_anno = read.csv('D:\\ONC/Annotations/BarkleyCanyonAnnotations_Public_Final.csv')
ONC_anno = read.csv('E:/DCLDE/ONC/Annotations/BarkleyCanyonAnnotations_Public_Final.csv')

# Only keep Oo data
ONC_anno = ONC_anno[ONC_anno$Species %in% c('Oo', 'Mn'),]

colnames(ONC_anno)[8]<-'origEcotype'


ONC_anno$ClassSpecies = ONC_anno$Species
ONC_anno$KW_certain = NA
ONC_anno$ClassSpecies[grepl("\\|", ONC_anno$Species)]= 'UndBio'
ONC_anno$ClassSpecies[grepl("Mn", ONC_anno$Species)]= 'HW'
ONC_anno$ClassSpecies[grepl("Oo", ONC_anno$Species)]= 'KW'
ONC_anno$ClassSpecies[grepl("\\|Oo", ONC_anno$Species)]= 'KW'
ONC_anno$ClassSpecies[ONC_anno$Species %in% c('BA', 'Bp', 'MY','OD', 'DE',
                                              'UN', 'NN', 'Pm', 'Bm','Gg','CE',
                                              'AN', 'Lo')]= 'UndBio'
# Exclude clicks and buzzes
ONC_anno= ONC_anno[!ONC_anno$Call.Type %in% c('BZ', 'CK'),]


# Add Ecotype 
ONC_anno$Ecotype<-NA
ONC_anno$Ecotype[ONC_anno$origEcotype == 'SRKW'] ='SRKW'
ONC_anno$Ecotype[ONC_anno$origEcotype == 'BKW'] ='BKW'
ONC_anno$Ecotype[ONC_anno$origEcotype == 'OKW'] ='OKW'

# Set the ecotype to NA
ONC_anno$Ecotype[ONC_anno$ClassSpecies== 'UndBio']= NA


ONC_anno$KW = ifelse(ONC_anno$ClassSpecies=='KW',1,0)

# Killer whales that were in combination with any other species are low confidence
ONC_anno$KW_certain[grepl("\\|Oo", ONC_anno$Species) & ONC_anno$ClassSpecies== 'KW']= 0
ONC_anno$KW_certain[ONC_anno$Species == 'Oo']= 1


ONC_anno$KW_certain =NA 
ONC_anno$KW_certain[ONC_anno$KW ==1 & 
                      grepl("\\|", ONC_anno$KW) == FALSE]= 1
ONC_anno$KW_certain[ONC_anno$KW ==1 & 
                      grepl("\\|", ONC_anno$Species) == TRUE] = 0


ONC_anno$FileBeginSec= as.numeric(ONC_anno$Left.time..sec.)
ONC_anno$FileEndSec= as.numeric(ONC_anno$Right.time..sec.)
ONC_anno$HighFreqHz= as.numeric(ONC_anno$Top.freq..Hz.)
ONC_anno$LowFreqHz= as.numeric(ONC_anno$Bottom.freq..Hz.)
ONC_anno$Dep='BarkLeyCanyon'

# There are a few typos in the original data set resulting in NA values. 
ONC_anno= ONC_anno[!is.na(ONC_anno$FileBeginSec),]


# Get time of the file
ONC_anno <- ONC_anno %>%
  mutate(filename = as.character(Soundfile),  # Make sure the column is treated as character
         UTC = as.POSIXct(sub(".*_(\\d{8}T\\d{6}\\.\\d{3}Z).*", "\\1", filename),
                          format = "%Y%m%dT%H%M%S.%OSZ",  tz = 'UTC'))

# Get the time of the annotation
ONC_anno$UTC = ONC_anno$UTC+ seconds(ONC_anno$FileBeginSec)

ONC_anno$Provider= 'ONC'
ONC_anno$AnnotationLevel = 'Call'
ONC_anno$dur = as.numeric(ONC_anno$FileEndSec)  - 
  as.numeric(ONC_anno$FileBeginSec)
ONC_anno$end_time = ONC_anno$UTC+ seconds(ONC_anno$dur)

# Sort and then identify overlaps
ONC_anno <- ONC_anno %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))

# There are typos in the datates and files fix them all manually
badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20130615T062356.061Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20130615T062356', 
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.061)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140402T032959.061Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140402T032959', 
                                  format="%Y%m%dT%H%M%S", tz="UTC")+seconds(0.061)

# Was this date hand entered? Appears to be 60 rather than 06
badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140701T605216.144Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140701T065216',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.144)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140702T054216.391Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140702T054216',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.391)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140804T1345424.361Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140804T1345424',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.361)+ONC_anno$FileBeginSec[badidx]


badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140903T231409.061Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140903T231409',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.061)+ONC_anno$FileBeginSec[badidx]


badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20141004T102246.170Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20141004T102246',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.170)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20141103T174455.370.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20141103T174455',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.370)+ONC_anno$FileBeginSec[badidx]


badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140104T140150.099Z-HPF.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140104T140150',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.099)+ONC_anno$FileBeginSec[badidx]



badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140202T131456.257Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140104T140150',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.099)+ONC_anno$FileBeginSec[badidx]



badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20141004T102246.170Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20141004T102246',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.170)+ONC_anno$FileBeginSec[badidx]


dayFolderPath = 'E:\\DCLDE\\ONC\\Audio\\BarkleyCanyon'
ONC_anno$FilePath = file.path(dayFolderPath,
                              format(ONC_anno$UTC-seconds(ONC_anno$FileBeginSec), "%Y%m%d"),
                              ONC_anno$Soundfile)
ONC_anno$AnnotationLevel = 'call'

ONC_anno$FileOk  = file.exists(ONC_anno$FilePath)

ONC_anno$FileUTC =ONC_anno$UTC- seconds(as.numeric(ONC_anno$FileBeginSec))


# Create a list of the 'not ok files'
missingData = ONC_anno[ONC_anno$FileOk== FALSE,]

# Remove the annotations for the missing files
ONC_anno = ONC_anno[!ONC_anno$Soundfile %in% missingData$Soundfile,]


ONC_anno= ONC_anno[,colOut]

rm(list= c('missingData', 'badidx', 'dayFolderPath'))

# Make sure all audio files are present for all annotations
if (all(ONC_anno$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


runTests(ONC_anno, EcotypeList, ClassSpeciesList)

#########################################################################

# No seconds in UTC
DFO_CRP1 = read.csv('E:/DCLDE/DFO_CRP/Annotations/annot_H50bjRcb_SM_det.csv')
DFO_CRP2 = read.csv('E:/DCLDE/DFO_CRP/Annotations/annot_KkHK0R2F_SM_det.csv')

DFO_CRP1$Dep='WVanIsl'
DFO_CRP2$Dep='NorthBc'


DFO_CRP = rbind(DFO_CRP1, DFO_CRP2)

table(DFO_CRP$sound_id_species)


DFO_CRP$ClassSpecies = DFO_CRP$sound_id_species

# Clean up the abiotic counds
DFO_CRP$ClassSpecies[
  DFO_CRP$ClassSpecies %in% c('Vessel Noise', 'Unknown', '', 'Mooring Noise', 
                              'Chain?', 'ADCP', 'Anchor Noise', 'Clang',
                              'Vessel Noise?', 'Chain','No sound data',
                              "Blast/Breach", "Fishing Gear", "Breach",
                              'Rubbing',"Anthro", "Nearby Uncharted Wreck",
                              'Water Noise', "Water Noise", "Mooring" ,
                              "Nothing","Mooring?", "Naval Sonar",
                              "Rocks?", "Rock?")] = 'AB'
# Esclude uncertain KW
DFO_CRP= DFO_CRP[!DFO_CRP$ClassSpecies %in% c("KW/PWSD?", "HW/KW?","KW?"),]


DFO_CRP$ClassSpecies[DFO_CRP$ClassSpecies %in% c("HW/GW", "HW/PWSD","HW/PWSD?",
                                                 "NPRW?HW?", 'HW?', "HW/GW?")] ='HW'

DFO_CRP$ClassSpecies[!DFO_CRP$ClassSpecies %in% ClassSpeciesList] = 'UndBio'
DFO_CRP$KW = ifelse(DFO_CRP$ClassSpecies == 'KW', 1,0)

# Set up the uncertainty
DFO_CRP$KW_certain= NA
DFO_CRP$KW_certain[DFO_CRP$KW==1] =1
DFO_CRP$KW_certain[DFO_CRP$sound_id_species %in% c("KW/PWSD?","HW/KW?",
                                                   "KW?")]=0


# Add Ecotype- note uncertain ecotypes getting their own guess
DFO_CRP$Ecotype = as.factor(DFO_CRP$kw_ecotype)
levels(DFO_CRP$Ecotype)<-c(NA, 'NRKW', 'NRKW', 'OKW', 'OKW', 'SRKW', 'SRKW',
                           'BKW', 'BKW', NA)
DFO_CRP$Ecotype[DFO_CRP$KW ==0]<-NA

# If the species is unsure then the ecotype is unsure
DFO_CRP$Ecotype[DFO_CRP$KW_certain==0]<- NA

# If the species is unsure then the ecotype is unsure
DFO_CRP$Ecotype[DFO_CRP$KW_certain==0]<- NA


DFO_CRP$Soundfile = DFO_CRP$filename
DFO_CRP$FileBeginSec = DFO_CRP$start
DFO_CRP$FileEndSec = DFO_CRP$end
DFO_CRP$LowFreqHz = DFO_CRP$freq_min
DFO_CRP$HighFreqHz = DFO_CRP$freq_max
DFO_CRP$UTC = as.POSIXct(DFO_CRP$utc, format="%Y-%m-%d %H:%M:%OS", tz = 'UTC')+
  seconds(as.numeric(DFO_CRP$utc_ms)/1000)
DFO_CRP$Provider = 'DFO_CRP'
DFO_CRP$AnnotationLevel = 'Call'

DFO_CRP$dur = DFO_CRP$FileEndSec  - DFO_CRP$FileBeginSec
DFO_CRP$end_time = DFO_CRP$UTC+ seconds(DFO_CRP$dur)

# Sort and then identify overlaps
DFO_CRP <- DFO_CRP %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))


dayFolderPath_WV = 'E:\\DCLDE\\DFO_CRP\\Audio\\DFOCRP_H50bjRcb-WCV1'
dayFolderPath_NBC = 'E:\\DCLDE\\DFO_CRP\\Audio\\DFOCRP_KkHK0R2F-NML1'

DFO_CRP$FilePath = 'blarg'

WVIidx = which(DFO_CRP$Dep == 'WVanIsl')
NBCidx = which(DFO_CRP$Dep != 'WVanIsl')

DFO_CRP$FilePath[WVIidx] = 
  file.path(dayFolderPath_WV, format(DFO_CRP$UTC[WVIidx], "%Y%m%d"),
            DFO_CRP$Soundfile[WVIidx])
DFO_CRP$FilePath[NBCidx] = 
  file.path(dayFolderPath_NBC, format(DFO_CRP$UTC[NBCidx], "%Y%m%d"),
            DFO_CRP$Soundfile[NBCidx])

DFO_CRP$FileOk  = file.exists(DFO_CRP$FilePath) 

DFO_CRP$dur = DFO_CRP$FileEndSec  - DFO_CRP$FileBeginSec

# # Sort and then identify overlaps
# DFO_CRP <- DFO_CRP %>%
#   arrange(Dep,UTC) %>%
#   mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))

rm(list= c('DFO_CRP1', 'DFO_CRP2'))

# Make sure all audio files are present for all annotations
if (all(DFO_CRP$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

runTests(DFO_CRP, EcotypeList, ClassSpeciesList)


# Exclude uncertain KW ecotype
DFO_CRP=subset(DFO_CRP, kw_ecotype %in% c("", "TKW", "NRKW", "OKW", 'SRKW'))

# Exclude KW without an ecotype
which(DFO_CRP$ClassSpecies== 'KW' & DFO_CRP$kw_ecotype =='')

DFO_CRP = DFO_CRP[, c(colOut)]

############################################################################
# DFO Yurk
############################################################################

# Get a list of files matching the pattern 'annot_Malahat'
file_list <- list.files(path = 'E:\\DCLDE\\DFO_WDLP/Annotations/merged_annotations/', 
                        pattern = '*csv', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
DFO_WDLP <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.csv(file)
  if (nrow(data) > 0) {
    data$Dep <- as.factor(basename(file))  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))

levels(DFO_WDLP$Dep)<-c('CarmanahPt', 'StrGeoN1', 'StrGeoN2','StrGeoS1',
                        'StrGeoS2','SwanChan')

# Fucking PAMGuard
DFO_WDLP = DFO_WDLP[DFO_WDLP$duration>0,]
DFO_WDLP = DFO_WDLP[!duplicated(DFO_WDLP),]

# Standardize formatting
DFO_WDLP$Soundfile = DFO_WDLP$soundfile
DFO_WDLP$LowFreqHz = DFO_WDLP$lf
DFO_WDLP$HighFreqHz = DFO_WDLP$hf
DFO_WDLP$UTC = as.POSIXct( DFO_WDLP$date_time_utc,  
                           format="%Y-%m-%d %H:%M:%OS", tz = 'UTC')

DFO_WDLP$FileBeginSec = DFO_WDLP$elapsed_time_seconds
DFO_WDLP$FileEndSec = DFO_WDLP$FileBeginSec+DFO_WDLP$duration/1000

DFO_WDLP$Ecotype = as.factor(DFO_WDLP$species)
levels(DFO_WDLP$Ecotype)<-c(NA, 'NRKW', 'SRKW', 'BKW', NA, NA)

DFO_WDLP$ClassSpecies = as.factor(DFO_WDLP$species)
levels(DFO_WDLP$ClassSpecies)<-c('HW', 'KW', 'KW', 'KW', 'UndBio', 'AB')

DFO_WDLP$KW = ifelse(DFO_WDLP$ClassSpecies== 'KW', 1,0)
DFO_WDLP$KW_certain = NA
DFO_WDLP$KW_certain[DFO_WDLP$KW==1]<-1
DFO_WDLP$Provider = 'DFO_WDA'


DFO_WDLP$AnnotationLevel = 'call'



DFO_WDLP$dur = DFO_WDLP$FileEndSec  - DFO_WDLP$FileBeginSec
DFO_WDLP$end_time = DFO_WDLP$UTC+ seconds(DFO_WDLP$dur)

# Sort and then identify overlaps
DFO_WDLP <- DFO_WDLP %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))

# Add a new column for deployment folder
DFO_WDLP$DepFolder = DFO_WDLP$Dep
levels(DFO_WDLP$DepFolder)<-c('CMN_2022-03-08_20220629_ST_utc',
                              'SOGN_20210905_20211129_AMAR_utc',
                              'SOGN_20210905_20211129_AMAR_utc',
                              'SOGS_20210904_20211118_AMAR_utc',
                              'SOGS_20210904_20211118_AMAR_utc',
                              'SWAN_20211113_20220110_AMAR_utc')

# Filepaths
dayFolderPath = 'E:\\DCLDE\\DFO_WDLP\\Audio'
DFO_WDLP$FilePath = 
  file.path(dayFolderPath, DFO_WDLP$DepFolder, 
            format(DFO_WDLP$UTC, "%Y%m%d"),
            DFO_WDLP$Soundfile)

DFO_WDLP$FileOk  = file.exists(DFO_WDLP$FilePath) 

# Make sure all audio files are present for all annotations
if (all(DFO_WDLP$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

runTests(DFO_WDLP, EcotypeList, ClassSpeciesList)
DFO_WDLP = DFO_WDLP[, colOut]
rm(list=c('dayFolderPath','dayFolderPath_NBC','dayFolderPath_WV','NBCidx','WVIidx'))


###########################################################################
# SIMRES
##########################################################################


# multiple selection tables
file_list <- list.files(path = 'e:\\DCLDE\\SIMRES/Annotations/', 
                        pattern = '*txt', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
SIMRES <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.table(file, header = TRUE, sep = '\t')
  if (nrow(data) > 0) {
    data$Dep <- basename(file)  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))

# Clean out blank rows
SIMRES = SIMRES[!is.na(SIMRES$Selection),]

SIMRES$ClassSpecies = as.factor(SIMRES$Sound.ID.Species)
levels(SIMRES$ClassSpecies)<-c('AB', 'AB','KW',  'KW', '')

# Cull uncertain sounds
SIMRES = SIMRES[SIMRES$ClassSpecies %in% c('KW'),]

#screw it only keep high certainty sounds
SIMRES = SIMRES[SIMRES$Confidence %in% c('high', 'High'),]
SIMRES = SIMRES[SIMRES$KW.Ecotype %in% c('SR'),]


#check SIMRES files in UTC
SIMRES$UTC = as.POSIXct(sub(".*_(\\d{8}T\\d{6}\\.\\d{3}Z)_.*", "\\1", 
                            SIMRES$Begin.File),  
                        format = "%Y%m%dT%H%M%S.%OSZ",
                        tz = 'UTC')+seconds(SIMRES$File.Offset..s.)

SIMRES$Ecotype = as.factor(SIMRES$KW.Ecotype)
levels(SIMRES$Ecotype)<- c(NA, NA, 'SRKW', 'SRKW')

SIMRES$KW =ifelse(SIMRES$ClassSpecies=='KW', 1, 0)

SIMRES$KW_certain = NA
SIMRES$KW_certain[SIMRES$KW ==1] =1
SIMRES$KW_certain[SIMRES$Confidence %in% c('low')]=0


SIMRES$Soundfile = SIMRES$Begin.File
SIMRES$Dep = 'Tekteksen'
SIMRES$LowFreqHz = SIMRES$Low.Freq..Hz.
SIMRES$HighFreqHz = SIMRES$Low.Freq..Hz.
SIMRES$FileBeginSec = SIMRES$File.Offset..s.
SIMRES$FileEndSec = SIMRES$File.Offset..s.+SIMRES$Delta.Time..s.
SIMRES$Provider = 'SIMRES'
SIMRES$AnnotationLevel = 'Call'


# Filepaths- wackadoodle for SIMRES
dayFolderPath = 'E:\\DCLDE\\SIMRES\\Audio\\'

# 1. List all files in the target directory
allAudio <- list.files(path = dayFolderPath, pattern = "\\.flac$", 
                       full.names = TRUE, recursive = TRUE)

# 2. Check if each file exists and get the full path
SIMRES$FilePath <- sapply(SIMRES$Soundfile, function(filename) {
  # Find the full path of the file (if it exists)
  full_path <- allAudio[grep(paste0("/", filename, "$"), allAudio)]
  if (length(full_path) > 0) return(full_path[1]) else return(NA)
})



SIMRES <- SIMRES %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(SIMRES$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


runTests(SIMRES, EcotypeList, ClassSpeciesList)

# For this data only keep 'high confidence kw' calls


SIMRES= SIMRES[, colOut]
rm(list= c('file_list', 'dayFolderPath'))



############################################################################
# VFPA - JASCO Strait of Georgia (Roberts Bank in Globus)
############################################################################

# Strait fo Georgia
VPFA_SoG<- read.csv('E:\\DCLDE\\VFPA/Annotations/annot_RB_man_det.csv')

VPFA_SoG <- VPFA_SoG %>%
  mutate(
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}.\\d{3}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )

# Get rid of anything with a question mark.
VPFA_SoG <- VPFA_SoG[!apply(VPFA_SoG, 1, function(row) any(grepl("\\?", row))), ]

# Get rid of medium or low confidence kw values values 
VPFA_SoG= VPFA_SoG[-c(which(VPFA_SoG$sound_id_species == 'KW' & 
                            VPFA_SoG$confidence != 'High')),]

VPFA_SoG$ClassSpecies =VPFA_SoG$sound_id_species

VPFA_SoG$ClassSpecies[VPFA_SoG$ClassSpecies %in% 
                        c("Vessel Noise", "Vessel Noise?",  "Noise", 
                          "Sonar","UN")]= 'AB'

VPFA_SoG$ClassSpecies[VPFA_SoG$ClassSpecies %in% 
                        c("PWSD","FS")]= 'UndBio'



VPFA_SoG$KW = ifelse(VPFA_SoG$ClassSpecies == 'KW',1,0)
VPFA_SoG$KW_certain= NA
VPFA_SoG$KW_certain[VPFA_SoG$KW==1] =1
VPFA_SoG$KW_certain[VPFA_SoG$KW==1 & 
                      grepl("\\?", VPFA_SoG$sound_id_species)]<-0



VPFA_SoG$UTC = VPFA_SoG$UTC+ seconds(as.numeric(VPFA_SoG$start))

# Add Ecotype 
VPFA_SoG$Ecotype = NA
VPFA_SoG$Ecotype[VPFA_SoG$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_SoG$Ecotype[VPFA_SoG$kw_ecotype == 'TKW'] ='BKW'


colnames(VPFA_SoG)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                    'FileEndSec', 'Soundfile')


VPFA_SoG$AnnotationLevel = 'Call'

VPFA_SoG$dur = VPFA_SoG$FileEndSec-VPFA_SoG$FileBeginSec

VPFA_SoG$end_time = VPFA_SoG$UTC+ seconds(VPFA_SoG$dur)

VPFA_SoG$Dep='StraitofGeorgia'
VPFA_SoG$Provider = 'JASCO_VFPA'

# Sort and then identify overlaps
VPFA_SoG <- VPFA_SoG %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))


# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files('E:\\DCLDE\\VFPA/StraitofGeorgia_Globus-RobertsBank/',
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)





# Day folder
dayFolderPath = 'E:\\DCLDE\\VFPA/Audio/StraitofGeorgia_Globus-RobertsBank/'
VPFA_SoG$FilePath = file.path(dayFolderPath,
                              format(VPFA_SoG$UTC-seconds(VPFA_SoG$FileBeginSec), 
                                     "%Y%m%d"),
                              VPFA_SoG$Soundfile)

VPFA_SoG <- VPFA_SoG %>%
  mutate(FileOk = file.exists(FilePath))


# Make sure all audio files are present for all annotations
if (all(VPFA_SoG$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}



VPFA_SoG =VPFA_SoG[,colOut]

runTests(VPFA_SoG, EcotypeList, ClassSpeciesList)


############################################################################
# VFPA - JASCO Boundary Pass
############################################################################


# Boundary Pass
VPFA_BoundaryPass<- read.csv('E:\\DCLDE\\VFPA/Annotations/annot_BP_man_det.csv')
VPFA_BoundaryPass <- VPFA_BoundaryPass %>%
  mutate(
    KW = as.numeric(grepl("KW", sound_id_species)),
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )



# Remove 'duplicate' and 'repeat' annotations
VPFA_BoundaryPass= VPFA_BoundaryPass[
  !VPFA_BoundaryPass$sound_id_species %in% c('Repeat', 'Duplicate'),]


# Get rid of anything with a question mark.
VPFA_BoundaryPass <- VPFA_BoundaryPass[!apply(VPFA_BoundaryPass, 1, 
                                              function(row) any(grepl("\\?", row))), ]

# only keep high confidence annotations
VPFA_BoundaryPass= VPFA_BoundaryPass[VPFA_BoundaryPass$confidence %in% 
                                       c("", 'High', "High & High", 
                                         "High & High & High",
                                         "High & High & High & High & High"),]


VPFA_BoundaryPass$ClassSpecies<- VPFA_BoundaryPass$sound_id_species


VPFA_BoundaryPass$ClassSpecies[VPFA_BoundaryPass$sound_id_species %in% 
                                 c("Vessel Noise", "Vessel Noise?",  "Noise", 
                                   "Sonar","UN", "NN", "BACKGROUND", 'UNK')]= 'AB'

VPFA_BoundaryPass$ClassSpecies[VPFA_BoundaryPass$sound_id_species %in% 
                                 c("PWSD","FS",  "PWSD?")]= 'UndBio'
VPFA_BoundaryPass$KW = ifelse(VPFA_BoundaryPass$ClassSpecies== 'KW', 1,0)
VPFA_BoundaryPass$KW_certain= NA
VPFA_BoundaryPass$KW_certain[VPFA_BoundaryPass$KW==1] =1
UncertainKWidx = which(VPFA_BoundaryPass$KW==1 & 
                         grepl("\\?", VPFA_BoundaryPass$sound_id_species))
VPFA_BoundaryPass$KW_certain[UncertainKWidx]=0

# Get time in UTC
VPFA_BoundaryPass$UTC = VPFA_BoundaryPass$UTC+ 
  seconds(as.numeric(VPFA_BoundaryPass$start))

# Add Ecotype 
VPFA_BoundaryPass$Ecotype = NA
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'TKW'] ='BKW'



colnames(VPFA_BoundaryPass)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                             'FileEndSec', 'Soundfile')



VPFA_BoundaryPass$AnnotationLevel = 'Call'
VPFA_BoundaryPass$dur = VPFA_BoundaryPass$FileEndSec-VPFA_BoundaryPass$FileBeginSec
VPFA_BoundaryPass$end_time = VPFA_BoundaryPass$UTC+ seconds(VPFA_BoundaryPass$dur)
VPFA_BoundaryPass$Dep='BoundaryPass'
VPFA_BoundaryPass$Provider = 'JASCO_VFPA'



# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files('E:\\DCLDE\\VFPA/Audio/BoundaryPass/',
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)


# Day folder
dayFolderPath = 'E:\\DCLDE\\VFPA/Audio/BoundaryPass//'
VPFA_BoundaryPass$FilePath = file.path(dayFolderPath, format(
  VPFA_BoundaryPass$UTC-seconds(VPFA_BoundaryPass$FileBeginSec),"%Y%m%d"),
  VPFA_BoundaryPass$Soundfile)

VPFA_BoundaryPass <- VPFA_BoundaryPass %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_BoundaryPass$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


VPFA_BoundaryPass = VPFA_BoundaryPass[,colOut]
runTests(VPFA_BoundaryPass, EcotypeList, ClassSpeciesList)


############################################################################
# VFPA - JASCO Haro Strait North
############################################################################


# Haro Strait North
VPFA_HaroNB<- read.csv('E:\\DCLDE\\VFPA/Annotations/annot_VFPA-HaroStrait-NB_SM_coarse.csv')
VPFA_HaroNB <- VPFA_HaroNB %>%
  mutate(
    KW = as.numeric(grepl("KW", sound_id_species)),
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )


# Get rid of anything with a question mark.
VPFA_HaroNB <- VPFA_HaroNB[!apply(VPFA_HaroNB, 1, 
                                              function(row) any(grepl("\\?", row))), ]

VPFA_HaroNB$ClassSpecies<- VPFA_HaroNB$sound_id_species


VPFA_HaroNB$ClassSpecies[VPFA_HaroNB$sound_id_species %in% 
                           c("Vessel Noise", "Vessel Noise?",  "Noise", 
                             "Sonar","UN", "BELL","VESSEL", "UNK")]= 'AB'

VPFA_HaroNB$ClassSpecies[VPFA_HaroNB$sound_id_species %in% 
                           c("PWSD","FS",  "PWSD?")]= 'UndBio'

VPFA_HaroNB$KW_certain= NA
VPFA_HaroNB$KW_certain[VPFA_HaroNB$KW==1] =1

UncertainKWidx = which(VPFA_HaroNB$KW==1 & 
                         grepl("\\?", VPFA_HaroNB$sound_id_species))
VPFA_HaroNB$KW_certain[UncertainKWidx]=0



# Get time in UTC
VPFA_HaroNB$UTC = VPFA_HaroNB$UTC+ 
  seconds(as.numeric(VPFA_HaroNB$start))

# Add Ecotype 
VPFA_HaroNB$Ecotype = NA
VPFA_HaroNB$Ecotype[VPFA_HaroNB$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_HaroNB$Ecotype[VPFA_HaroNB$kw_ecotype == 'TKW'] ='BKW'



colnames(VPFA_HaroNB)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                       'FileEndSec', 'Soundfile')



VPFA_HaroNB$AnnotationLevel = 'Call'
VPFA_HaroNB$dur = VPFA_HaroNB$FileEndSec-VPFA_HaroNB$FileBeginSec
VPFA_HaroNB$end_time = VPFA_HaroNB$UTC+ seconds(VPFA_HaroNB$dur)
VPFA_HaroNB$Dep='HaroStraitNorth'
VPFA_HaroNB$Provider = 'JASCO_VFPA'


# Day folder
dayFolderPath = 'E:\\DCLDE\\VFPA/Audio/VFPA-HaroStrait-NB///'
VPFA_HaroNB$FilePath = file.path(dayFolderPath, format(
  VPFA_HaroNB$UTC-seconds(VPFA_HaroNB$FileBeginSec),"%Y%m%d"),
  VPFA_HaroNB$Soundfile)

VPFA_HaroNB <- VPFA_HaroNB %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_HaroNB$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}



VPFA_HaroNB= VPFA_HaroNB[,colOut]
runTests(VPFA_HaroNB, EcotypeList, ClassSpeciesList)


############################################################################
# VFPA - JASCO Haro Strait South
############################################################################

# Haro Strait South
VPFA_HaroSB<- read.csv('E:\\DCLDE\\VFPA/Annotations/annot_VFPA-HaroStrait-SB_SM_coarse.csv')
VPFA_HaroSB <- VPFA_HaroSB %>%
  mutate(
    KW = as.numeric(grepl("KW", sound_id_species)),
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )


# Get rid of anything with a question mark.
VPFA_HaroSB <- VPFA_HaroSB[!apply(VPFA_HaroSB, 1, 
                                  function(row) any(grepl("\\?", row))), ]


VPFA_HaroSB$ClassSpecies<- VPFA_HaroSB$sound_id_species


VPFA_HaroSB$ClassSpecies[VPFA_HaroSB$sound_id_species %in% 
                           c("Vessel Noise", "Vessel Noise?",  "Noise", 
                             "Sonar","UN", "BELL","VESSEL", "UNK")]= 'AB'

VPFA_HaroSB$ClassSpecies[VPFA_HaroSB$sound_id_species %in% 
                           c("PWSD","FS",  "PWSD?")]= 'UndBio'

VPFA_HaroSB$KW_certain= NA
VPFA_HaroSB$KW_certain[VPFA_HaroSB$KW==1] =1


# Get time in UTC
VPFA_HaroSB$UTC = VPFA_HaroSB$UTC+ 
  seconds(as.numeric(VPFA_HaroSB$start))

# Add Ecotype 
VPFA_HaroSB$Ecotype = NA
VPFA_HaroSB$Ecotype[VPFA_HaroSB$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_HaroSB$Ecotype[VPFA_HaroSB$kw_ecotype == 'TKW'] ='BKW'



colnames(VPFA_HaroSB)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                       'FileEndSec', 'Soundfile')


VPFA_HaroSB$AnnotationLevel = 'Call'
VPFA_HaroSB$dur = VPFA_HaroSB$FileEndSec-VPFA_HaroSB$FileBeginSec
VPFA_HaroSB$end_time = VPFA_HaroSB$UTC+ seconds(VPFA_HaroSB$dur)
VPFA_HaroSB$Dep='HaroStraitSouth'
VPFA_HaroSB$Provider = 'JASCO_VFPA'

# Day folder
dayFolderPath = 'E:\\DCLDE\\VFPA/Audio/VFPA-HaroStrait-SB/'
VPFA_HaroSB$FilePath = file.path(dayFolderPath, format(
  VPFA_HaroSB$UTC-seconds(VPFA_HaroSB$FileBeginSec),"%Y%m%d"),
  VPFA_HaroSB$Soundfile)

VPFA_HaroSB <- VPFA_HaroSB %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_HaroSB$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}



VPFA_HaroSB= VPFA_HaroSB[, colOut]
runTests(VPFA_HaroSB, EcotypeList, ClassSpeciesList)



#############################################################################
# SMRU Consulting
#############################################################################

# Strait fo Georgia
SMRU_SRKW<- read.csv('E:\\DCLDE\\SMRU/annotations/annot_LimeKiln-Encounters_man_det.csv')
SMRU_HW<- read.csv('E:\\DCLDE\\SMRU/annotations/annot_LimeKiln-Humpback_man_det.csv')

SMRU <- rbind(SMRU_HW, SMRU_SRKW)

SMRU <- SMRU %>%
  mutate(
    UTC = as.POSIXct(sub(".*_(\\d{8}_\\d{6}_\\d{3})\\..*", "\\1", filename), 
                     format = "%Y%m%d_%H%M%S_%OS", tz = "UTC")
  )

# Some annotations witout sound species ID, remove
SMRU = SMRU[SMRU$sound_id_species != "",]

# Get rid of anything with a question mark.
SMRU <- SMRU[!apply(SMRU, 1,   function(row) any(grepl("\\?", row))), ]



SMRU$ClassSpecies=SMRU$sound_id_species
SMRU$ClassSpecies[SMRU$sound_id_species %in% 
                    c('KW?', "KW", "HW/KW?", "KW/PWSD","KW/PWSD?",
                      "PWSD/KW?")]<-'KW'

SMRU$ClassSpecies[SMRU$ClassSpecies %in% 
                    c("Vessel Noise", "Vessel Noise?",  "Noise", 
                      "Sonar","UN", "Unknown", "Mooring", "surface noise",
                      "Vessel noise", "Unknown?", "Mooring noise", 
                      "Wave noise", "Vessel  Noise", "Surf noise",
                      "Waves", "Mooring ", "Mooring?")]= 'AB'

SMRU$ClassSpecies[SMRU$ClassSpecies %in% 
                    c("PWSD","FS", "Seal?", "Fish", "Fish?", "FISH",
                      "Snapping shrimp/urchin", 
                      "Snapping shrimp or urchin" )]= 'UndBio'


# Set the KW class and certainty
SMRU$KW = ifelse(SMRU$ClassSpecies == 'KW',1,0)
SMRU$KW_certain= NA
SMRU$KW_certain[SMRU$KW==1] =1
SMRU$KW_certain[SMRU$KW==1 & 
                  grepl("\\?", SMRU$sound_id_species)]<-0

SMRU$UTC = SMRU$UTC+ seconds(as.numeric(SMRU$start))

# Add Ecotype 
SMRU$Ecotype = NA
SMRU$Ecotype[SMRU$kw_ecotype == 'SRKW'] ='SRKW'



colnames(SMRU)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                'FileEndSec', 'Soundfile')


SMRU$AnnotationLevel = 'Call'

SMRU$dur = SMRU$FileEndSec-SMRU$FileBeginSec
SMRU$end_time = SMRU$UTC+ seconds(SMRU$dur)

SMRU$Dep='LimeKiln'
SMRU$Provider = 'SMRUConsulting'

# Sort and then identify overlaps
SMRU <- SMRU %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))



# Day folder
dayFolderPath = 'E:\\DCLDE\\SMRU/Audio/Lime Kiln/'
SMRU$FilePath = file.path(dayFolderPath,
                          SMRU$path)

SMRU <- SMRU %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(SMRU$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

# Exclude uncertain calls
SMRU= SMRU[!SMRU$sound_id_species 
                         %in% c("HW/KW?", "KW/HW?","KW/PWSD?", "KW?", 'PSWD?', 'KW/PWSD',
                                "PWSD/KW?", "KW/PWSD?", "HW/KW?" ),]
SMRU= SMRU[!SMRU$kw_ecotype %in% c("SRKW?","TKW?", 'Unknown'),]

# Exclude low and medium confident calls
SMRU= SMRU[!SMRU$confidence %in% c('low', 'M', 'Medium', 'Low', 'L', 'HM'),]

SMRU =SMRU[,colOut]

runTests(SMRU, EcotypeList, ClassSpeciesList)

###########################################################################
allAnno = rbind(DFO_WDLP, ONC_anno, SIMRES,
                VPFA_BoundaryPass, VPFA_HaroNB, VPFA_HaroSB, VPFA_SoG, SMRU)

#kick out any killer whale annotations without an ecotype
allAnno= allAnno[-c(which(is.na(allAnno$Ecotype) & 
                            allAnno$ClassSpecies == 'KW')),]

# These will be exported as folders based on names
allAnno$AnnoBin = as.factor(paste0(allAnno$ClassSpecies, '_', allAnno$Ecotype))

levels(allAnno$AnnoBin)<- c('Background', 
                            'Megaptera novaanglea_Humpback Whale',
                            'Orcinus orcaBKW_Biggs Killerwhale', 
                            'Orcinus orcaNRKW_NorthernRes Killerwhale', 
                            'Orcinus orcaOKW_Offshore Killerwhale',
                            'Orcinus orcasSRKW_SouthernRes Killerwhale',
                            'UndetermedBio')
allAnno
# Export the data as a CSV
write.csv(allAnno, 'PNW_ClassifierAnnotations.csv')


