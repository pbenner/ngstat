
ngstat exec peakCaller.so LearnModel peakCaller.json /project/wig-data/mouse-embryo/H3K27ac-mm10-liver-day12.5.raw.bw
ngstat exec ../../plugins/nonparametric/nonparametric.so Estimate 500 /project/wig-data/mouse-embryo/H3K27ac-mm10-liver-day12.5.raw.bw peakCaller-nonparametric.json

ngstat exec peakCaller.so CallPeaks result.bw /project/wig-data/mouse-embryo/H3K27ac-mm10-liver-day12.5.raw.bw peakCaller.json

~/Source/gonetics/tools/bigWigPositive/bigWigPositive -v result.table result.bw:0.95

