
TRACK1=../data/Liver-Day12.5-H3K27ac.raw.bw
TRACK2=../data/Liver-Day12.5-Control.raw.bw

# estimate treatment mixture model
ngstat exec peakCaller.so LearnModel peakCaller-treatment.json $TRACK1
# estimate control mixture model
ngstat exec peakCaller.so LearnModel peakCaller-control.json $TRACK2
# use estimates models to call peaks
ngstat exec peakCaller.so CallPeaks --model-treatment=peakCaller-treatment.json --model-control=peakCaller-control.json result.bw $TRACK1 $TRACK2

# extract positive regions from resulting track
bigWigPositive -v result.table result.bw:0.95
