
TRACK=../data/Liver-Day12.5-H3K27ac.raw.bw

# estimate mixture model
ngstat exec peakCaller.so LearnModel peakCaller.json $TRACK
# use estimates model to call peaks
ngstat exec peakCaller.so CallPeaks --model=peakCaller.json result.bw $TRACK

# extract positive regions from resulting track
bigWigPositive -v result.table result.bw:0.95
