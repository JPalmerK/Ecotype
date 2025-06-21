# First birdnet will be a balanced datset across the labels with 3k each of
# the ecotype labels. No attention will be paid to call types

# make it repeatable
set.seed(5)
DCLDE_train = read.csv( 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv')


labels = unique(DCLDE_train$Labels)
trainIDX = c()


for(label in labels){
  labelIdx = which(DCLDE_train$Labels == label) 
  
  if (length(labelIdx)<3000)
  {idx = sample(labelIdx, 3000, replace = TRUE)}
  else{idx = sample(labelIdx, 3000, replace = FALSE)}
  trainIDX= c(trainIDX, idx)
  
  
}

# Write the database
DCLDE_train_birdnet01 = DCLDE_train[trainIDX,]

# Exclude alaska and offshore data
DCLDE_train_birdnet01= DCLDE_train_birdnet01[DCLDE_train_birdnet01$Labels %in% 
                                               c('Background', 'HW', 'SRKW','TKW'),]

table(DCLDE_train_birdnet01$Labels)

write.csv(DCLDE_train_birdnet01, 'DCLDE_train_birdnet01.csv')