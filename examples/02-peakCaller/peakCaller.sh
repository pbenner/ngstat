
TRACK1=../data/Liver-Day12.5-H3K27ac.raw.bw
TRACK2=../data/Liver-Day12.5-Control.raw.bw

# estimate treatment mixture model
ngstat exec peakCaller.so LearnModel treatment peakCaller-treatment.json $TRACK1
# estimate control mixture model
ngstat exec peakCaller.so LearnModel control peakCaller-control.json $TRACK2
# use estimates models to call peaks
ngstat exec peakCaller.so CallPeaks --model-treatment=peakCaller-treatment.json --model-control=peakCaller-control.json result.bw $TRACK1 $TRACK2

# extract positive regions from resulting track
bigWigPositive -v result.table result.bw:0.95


# get control peaks
ngstat exec ../01-peakCaller/peakCaller.so CallPeaks --model peakCaller-control.json --components 8 result-control.bw $TRACK2
bigWigPositive -v result-control.table result-control.bw:0.95


# good example: chr6:29731448-29776447
